import { useState } from 'react';
import { Pressable, ScrollView, Text, TextInput, View } from 'react-native';
import type { Language } from '../core/types';
import { MODEL } from '../ml/detector';
import { saveSettings, type Settings } from '../store/storage';
import { s } from '../ui/theme';

const LANGS: [Language, string][] = [['en', 'English'], ['fr', 'Francais'], ['rw', 'Kinyarwanda']];

export function SettingsScreen({ settings, onSaved }: { settings: Settings; onSaved: (s: Settings) => void }) {
  const [v, setV] = useState(settings);
  const [saved, setSaved] = useState(false);
  const set = (k: keyof Settings) => (t: string) => { setV({ ...v, [k]: t.trim() }); setSaved(false); };
  return (
    <ScrollView contentContainerStyle={s.screen}>
      <View style={s.card}>
        <Text style={s.h2}>Server connection</Text>
        <Text style={s.label}>Vision API (server model)</Text>
        <TextInput style={s.input} autoCapitalize="none" autoCorrect={false} value={v.visionUrl} onChangeText={set('visionUrl')} />
        <Text style={s.label}>Report API (Qwen)</Text>
        <TextInput style={s.input} autoCapitalize="none" autoCorrect={false} value={v.llmUrl} onChangeText={set('llmUrl')} />
        <Text style={s.label}>Access token</Text>
        <TextInput style={s.input} autoCapitalize="none" autoCorrect={false} secureTextEntry value={v.token} onChangeText={set('token')} />
        <Text style={s.label}>Report language</Text>
        <View style={s.row}>
          {LANGS.map(([k, name]) => (
            <Pressable key={k} style={[s.btn, k === v.language ? {} : s.btn2, { padding: 8 }]} onPress={() => { setV({ ...v, language: k }); setSaved(false); }}>
              <Text style={k === v.language ? s.btnT : s.btnT2}>{name}</Text>
            </Pressable>
          ))}
        </View>
        <View style={s.row}>
          <Pressable style={s.btn} onPress={async () => { await saveSettings(v); onSaved(v); setSaved(true); }}><Text style={s.btnT}>{saved ? 'Saved' : 'Save'}</Text></Pressable>
        </View>
      </View>
      <View style={s.card}>
        <Text style={s.h2}>On-device model</Text>
        <Text style={s.li}>{MODEL.installed ? `Installed: ${MODEL.version ?? 'unknown version'}` : 'No model bundled in this build.'}</Text>
        <Text style={s.muted}>Used automatically when the Vision API cannot be reached. Photos taken that way are re-checked by the server model when you are back online.</Text>
      </View>
    </ScrollView>
  );
}
