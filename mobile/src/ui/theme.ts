import { StyleSheet } from 'react-native';

export const C = { bg: '#f4f6f3', card: '#fff', ink: '#14201a', muted: '#627067', line: '#e3e9e2', brand: '#16794c', bad: '#b3261e', badBg: '#fde8e6', warn: '#a15c00', warnBg: '#fff3dc', ok: '#12713f', okBg: '#e6f2ea' };

export const s = StyleSheet.create({
  screen: { padding: 16, paddingBottom: 40 },
  card: { backgroundColor: C.card, borderRadius: 14, padding: 16, marginTop: 14, borderWidth: 1, borderColor: C.line },
  h2: { fontSize: 18, fontWeight: '800', color: C.ink },
  h3: { fontSize: 12, fontWeight: '700', color: C.muted, textTransform: 'uppercase', letterSpacing: 0.6, marginTop: 12 },
  li: { fontSize: 14, color: C.ink, marginTop: 3, lineHeight: 20 },
  muted: { fontSize: 13, color: C.muted, marginTop: 6 },
  row: { flexDirection: 'row', gap: 10, marginTop: 10, alignItems: 'center' },
  btn: { flex: 1, backgroundColor: C.brand, borderRadius: 10, padding: 13, alignItems: 'center' },
  btn2: { backgroundColor: C.card, borderWidth: 1, borderColor: C.line },
  btnT: { color: '#fff', fontWeight: '700' },
  btnT2: { color: C.brand, fontWeight: '700' },
  input: { borderWidth: 1, borderColor: C.line, borderRadius: 10, padding: 10, backgroundColor: C.bg, color: C.ink, flex: 1 },
  label: { fontSize: 12, fontWeight: '600', color: C.muted, marginTop: 10 },
  banner: { borderRadius: 10, padding: 12, marginTop: 10 },
  chip: { alignSelf: 'flex-start', paddingHorizontal: 10, paddingVertical: 3, borderRadius: 99, overflow: 'hidden', fontWeight: '700', fontSize: 12 },
});
