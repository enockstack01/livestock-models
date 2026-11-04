import * as ImagePicker from 'expo-image-picker';
import { useState } from 'react';
import { ActivityIndicator, Image, Pressable, ScrollView, StyleSheet, Text, TextInput, View } from 'react-native';
import { http } from '../api/http';
import { runScan, type ScanResult } from '../core/scan';
import type { AdvicePack, Language } from '../core/types';
import advice from '../data/diseases.json';
import { MODEL } from '../ml/detector';
import { onDevice } from '../ml/offline';
import { queueStore, type Settings } from '../store/storage';
import { ReportCard } from '../ui/ReportCard';
import { C, s } from '../ui/theme';

const PACK = advice as unknown as AdvicePack;
const COLORS = ['#e53935', '#1e88e5', '#fb8c00', '#8e24aa', '#00897b', '#6d4c41'];
const LANGS: [Language, string][] = [['en', 'English'], ['fr', 'Francais'], ['rw', 'Kinyarwanda']];
const newId = () => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;

export function ScanScreen({ settings, onQueueChanged, pending }: { settings: Settings; onQueueChanged: () => void; pending: number }) {
  const [photo, setPhoto] = useState<{ uri: string; width: number; height: number } | null>(null);
  const [res, setRes] = useState<ScanResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lang, setLang] = useState<Language>(settings.language);
  const [weight, setWeight] = useState('');
  const [district, setDistrict] = useState('');
  const [animalId, setAnimalId] = useState('');
  const [box, setBox] = useState({ w: 0, h: 0 });

  const scan = async (a: ImagePicker.ImagePickerAsset) => {
    setPhoto({ uri: a.uri, width: a.width, height: a.height }); setRes(null); setErr(null); setBusy(true);
    try {
      const w = Number(weight);
      const r = await runScan({ uri: a.uri, width: a.width, height: a.height },
        { language: lang, species: '', district, animalId, animal: w > 0 ? { weight_kg: w } : undefined },
        { visionUrl: settings.visionUrl, llmUrl: settings.llmUrl, token: settings.token },
        { http, onDevice, pack: PACK, queue: queueStore, now: Date.now, id: newId });
      setRes(r);
      if (r.queuedId) onQueueChanged();
    } catch (e) { setErr(e instanceof Error ? e.message : String(e)); }
    setBusy(false);
  };

  const pick = async (camera: boolean) => {
    const perm = camera ? await ImagePicker.requestCameraPermissionsAsync() : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) { setErr('Permission denied.'); return; }
    const r = camera ? await ImagePicker.launchCameraAsync({ quality: 1 }) : await ImagePicker.launchImageLibraryAsync({ mediaTypes: ['images'], quality: 1 });
    if (!r.canceled && r.assets[0]) await scan(r.assets[0]);
  };

  // boxes are in the analysed frame's pixels (the server's processed image online, the on-device frame offline); the photo is shown "contain"
  const frame = res?.image ?? (photo ? { width: photo.width, height: photo.height } : null);
  const fit = frame && box.w ? Math.min(box.w / frame.width, box.h / frame.height) : 0;
  const offX = frame ? (box.w - frame.width * fit) / 2 : 0, offY = frame ? (box.h - frame.height * fit) / 2 : 0;

  return (
    <ScrollView contentContainerStyle={s.screen}>
      <View style={s.row}>
        <Pressable style={s.btn} onPress={() => pick(true)}><Text style={s.btnT}>Take photo</Text></Pressable>
        <Pressable style={[s.btn, s.btn2]} onPress={() => pick(false)}><Text style={s.btnT2}>Choose photo</Text></Pressable>
      </View>
      <View style={s.row}>
        <TextInput style={s.input} placeholder="Weight kg (for doses)" keyboardType="numeric" value={weight} onChangeText={setWeight} />
        <TextInput style={s.input} placeholder="District" value={district} onChangeText={setDistrict} />
      </View>
      <View style={s.row}>
        <TextInput style={s.input} placeholder="Animal ID / tag (optional)" value={animalId} onChangeText={setAnimalId} />
      </View>
      <View style={s.row}>
        {LANGS.map(([k, name]) => (
          <Pressable key={k} style={[s.btn, k === lang ? {} : s.btn2, { padding: 8 }]} onPress={() => setLang(k)}><Text style={k === lang ? s.btnT : s.btnT2}>{name}</Text></Pressable>
        ))}
      </View>
      {pending > 0 && <Text style={s.muted}>{pending} photo(s) waiting to be confirmed online.</Text>}
      {MODEL.installed && !!MODEL.warning && <View style={[s.banner, { backgroundColor: C.warnBg }]}><Text style={{ color: C.warn }}>{MODEL.warning}</Text></View>}

      {photo && (
        <View style={styles.imgBox} onLayout={(e) => setBox({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height })}>
          <Image source={{ uri: photo.uri }} style={StyleSheet.absoluteFill} resizeMode="contain" />
          {res?.detections.map((d, i) => (
            <View key={i} style={[styles.det, { borderColor: COLORS[i % 6], left: offX + d.box[0] * fit, top: offY + d.box[1] * fit, width: (d.box[2] - d.box[0]) * fit, height: (d.box[3] - d.box[1]) * fit }]}>
              <Text style={[styles.detT, { backgroundColor: COLORS[i % 6] }]}>{d.class.replace(/_/g, ' ')} {Math.round(d.confidence * 100)}%</Text>
            </View>
          ))}
        </View>
      )}
      {busy && <ActivityIndicator style={{ margin: 16 }} />}
      {err && <View style={[s.banner, { backgroundColor: C.warnBg }]}><Text style={{ color: C.warn }}>{err}</Text></View>}

      {res && (
        <View>
          <View style={[s.banner, { backgroundColor: res.confirmed ? C.okBg : C.warnBg }]}>
            <Text style={{ color: res.confirmed ? C.ok : C.warn, fontWeight: '700' }}>
              {res.confirmed ? `Confirmed by ${res.modelVersion} (online)` : `Preliminary (offline): ${res.modelVersion} on this phone`}
            </Text>
            {!res.confirmed && <Text style={{ color: C.warn, marginTop: 4 }}>Saved. It will be confirmed by the server model automatically when you are back online.</Text>}
          </View>
          {res.flags.includes('possible_notifiable_disease') && !res.report.notifiable && <Text style={s.muted}>Possible notifiable disease flagged.</Text>}
          {res.detections.length === 0 && <Text style={s.muted}>Nothing recognised in this photo. That does not prove the animal is healthy: retake a closer, well-lit photo, or ask a vet.</Text>}
          <ReportCard report={res.report} source={res.reportSource === 'llm' ? 'llm' : 'pack'} notes={res.notes} />
        </View>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  imgBox: { height: 320, marginTop: 14, backgroundColor: '#e3e9e2', borderRadius: 12, overflow: 'hidden' },
  det: { position: 'absolute', borderWidth: 2, borderRadius: 4 },
  detT: { color: '#fff', fontSize: 11, fontWeight: '700', alignSelf: 'flex-start', paddingHorizontal: 4 },
});
