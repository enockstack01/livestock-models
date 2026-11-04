import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { base64ToBytes, rgbaToTensor } from '../src/core/preprocess.ts';
import { decodeYolo, iou, letterboxParams, nms, toOriginal } from '../src/core/yolo.ts';

/** Build a raw YOLO output with the given detections, in either layout. */
function makeOutput(dets: { cx: number; cy: number; w: number; h: number; cls: number; score: number }[], nc: number, N: number, layout: 'channels_first' | 'channels_last') {
  const C = 4 + nc;
  const out = new Float32Array(C * N);
  const set = (a: number, ch: number, v: number) => { out[layout === 'channels_first' ? ch * N + a : a * C + ch] = v; };
  dets.forEach((d, a) => { set(a, 0, d.cx); set(a, 1, d.cy); set(a, 2, d.w); set(a, 3, d.h); set(a, 4 + d.cls, d.score); });
  return out;
}
const DETS = [
  { cx: 320, cy: 320, w: 100, h: 100, cls: 1, score: 0.9 },
  { cx: 324, cy: 322, w: 100, h: 100, cls: 1, score: 0.6 }, // duplicate of the first (same class) -> suppressed
  { cx: 324, cy: 322, w: 100, h: 100, cls: 0, score: 0.7 }, // overlapping box, DIFFERENT class -> kept
  { cx: 100, cy: 500, w: 40, h: 40, cls: 2, score: 0.1 },   // below threshold
];

for (const layout of ['channels_first', 'channels_last'] as const) {
  test(`decode + per-class NMS (${layout})`, () => {
    const r = decodeYolo(makeOutput(DETS, 3, 8400, layout), { numClasses: 3, imgsz: 640, layout });
    assert.equal(r.length, 2);
    assert.deepEqual(r.map((d) => [d.classId, d.score.toFixed(2)]), [[1, '0.90'], [0, '0.70']]);
    assert.ok(Math.abs(r[0].box.x1 - 270) < 1e-3 && Math.abs(r[0].box.y2 - 370) < 1e-3);
  });
}

test('normalised (TFLite-style) coordinates are scaled to pixels', () => {
  const n = DETS.map((d) => ({ ...d, cx: d.cx / 640, cy: d.cy / 640, w: d.w / 640, h: d.h / 640 }));
  const r = decodeYolo(makeOutput(n, 3, 8400, 'channels_last'), { numClasses: 3, imgsz: 640, layout: 'channels_last' });
  assert.ok(Math.abs(r[0].box.x1 - 270) < 1e-2 && Math.abs(r[0].box.x2 - 370) < 1e-2);
});

test('wrong class count is rejected rather than mis-decoded', () => {
  assert.throws(() => decodeYolo(new Float32Array(7 * 100 + 1), { numClasses: 3, imgsz: 640, layout: 'channels_first' }), /multiple/);
});

test('lower confidence threshold raises recall', () => {
  const r = decodeYolo(makeOutput(DETS, 3, 100, 'channels_first'), { numClasses: 3, imgsz: 640, layout: 'channels_first', confThreshold: 0.05 });
  assert.equal(r.length, 3);
});

test('iou and nms basics', () => {
  const a = { x1: 0, y1: 0, x2: 10, y2: 10 };
  assert.equal(iou(a, a), 1);
  assert.equal(iou(a, { x1: 20, y1: 20, x2: 30, y2: 30 }), 0);
  assert.equal(nms([{ classId: 0, score: 1, box: a }, { classId: 0, score: 0.5, box: a }], 0.5, 10).length, 1);
  assert.equal(nms([{ classId: 0, score: 1, box: a }, { classId: 0, score: 0.5, box: { x1: 30, y1: 0, x2: 40, y2: 10 } }], 0.5, 1).length, 1); // maxDet
});

test('letterbox maths round-trips a box back to original pixels', () => {
  // 1280x720 photo into 640x640: scale 0.5, content 640x360, vertical pad 140
  const lb = letterboxParams(1280, 720, 640);
  assert.deepEqual([lb.scale, lb.padX, lb.padY, lb.newW, lb.newH], [0.5, 0, 140, 640, 360]);
  const orig = { x1: 200, y1: 100, x2: 600, y2: 500 };
  const inModel = { classId: 0, score: 0.9, box: { x1: orig.x1 * 0.5, y1: orig.y1 * 0.5 + 140, x2: orig.x2 * 0.5, y2: orig.y2 * 0.5 + 140 } };
  const [d] = toOriginal([inModel], lb, 1280, 720, ['lesion']);
  assert.deepEqual(d.box, orig);
  assert.equal(d.label, 'lesion');
  const clipped = toOriginal([{ classId: 5, score: 1, box: { x1: -50, y1: 0, x2: 700, y2: 700 } }], lb, 1280, 720, ['a']);
  assert.deepEqual([clipped[0].box.x1, clipped[0].box.x2, clipped[0].box.y2, clipped[0].label], [0, 1280, 720, 'class_5']);
});

test('preprocess: letterbox padding, normalisation and colour placement', () => {
  // 2x1 image: left pixel pure red, right pixel pure blue -> 8x8 model input
  const rgba = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);
  const { input, letterbox } = rgbaToTensor(rgba, 2, 1, 8);
  assert.equal(input.length, 8 * 8 * 3);
  assert.equal(letterbox.padY, 2);                                                    // 8x4 content centred in 8x8
  assert.ok(Math.abs(input[0] - 114 / 255) < 1e-6);                                  // padded row = grey 114
  const px = (x: number, y: number) => [0, 1, 2].map((c) => input[(y * 8 + x) * 3 + c]);
  const left = px(0, 2);
  const right = px(7, 2);
  assert.ok(left[0] > 0.9 && left[2] < 0.1, `left should be red-ish: ${left}`);
  assert.ok(right[2] > 0.9 && right[0] < 0.1, `right should be blue-ish: ${right}`);
  assert.ok(input.every((v) => v >= 0 && v <= 1));
});

test('base64 decoding handles padding and data URIs', () => {
  assert.deepEqual([...base64ToBytes('TWE=')], [77, 97]);
  assert.deepEqual([...base64ToBytes('TQ==')], [77]);
  assert.deepEqual([...base64ToBytes('data:image/jpeg;base64,TWFu')], [77, 97, 110]);
  const bytes = new Uint8Array(1000).map((_, i) => (i * 37) % 256);
  assert.deepEqual(base64ToBytes(Buffer.from(bytes).toString('base64')), bytes);
});

test('decodes real YOLO11n ONNX output exactly like ultralytics NMS (fixtures from a trained model)', () => {
  const dir = new URL('./fixtures/yolo11n_onnx/', import.meta.url);
  const ref = JSON.parse(readFileSync(new URL('reference.json', dir), 'utf8'));
  for (const [tag, expected] of Object.entries<any[]>(ref.cases)) {
    const b = readFileSync(new URL(`${tag}.bin`, dir));
    const raw = new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
    assert.equal(raw.length, 7 * 8400);
    const mine = decodeYolo(raw, { numClasses: ref.names.length, imgsz: ref.size, layout: 'channels_first', confThreshold: ref.conf, iouThreshold: ref.iou });
    assert.equal(mine.length, expected.length, `${tag}: detection count`);            // includes the 'nothing detected' case
    for (const e of expected) {
      const m = mine.find((d) => ref.names[d.classId] === e.class)!;
      assert.ok(m, `${tag}: missing ${e.class}`);
      assert.ok(Math.abs(m.score - e.conf) < 0.002, `${tag}: score ${m.score} vs ${e.conf}`);
      assert.ok(iou(m.box, { x1: e.box[0], y1: e.box[1], x2: e.box[2], y2: e.box[3] }) > 0.99, `${tag}: box`);
    }
  }
});

test('preprocess: planar NCHW layout puts each colour channel in its own plane (what the LiteRT export expects)', () => {
  const rgba = new Uint8Array([255, 0, 0, 255, 0, 0, 255, 255]);               // 2x1: red | blue
  const hwc = rgbaToTensor(rgba, 2, 1, 8, 'nhwc').input;
  const chw = rgbaToTensor(rgba, 2, 1, 8, 'nchw').input;
  assert.equal(chw.length, 3 * 8 * 8);
  const plane = 64;
  for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) for (let c = 0; c < 3; c++)
    assert.equal(chw[c * plane + y * 8 + x], hwc[(y * 8 + x) * 3 + c]);          // identical values, different memory order
  assert.ok(chw[0 * plane + 2 * 8 + 0] > 0.9 && chw[2 * plane + 2 * 8 + 0] < 0.1);   // left pixel: R plane high, B plane low
  assert.ok(chw[2 * plane + 2 * 8 + 7] > 0.9 && chw[0 * plane + 2 * 8 + 7] < 0.1);   // right pixel: B plane high, R plane low
  assert.ok(Math.abs(chw[0] - 114 / 255) < 1e-6 && Math.abs(chw[2 * plane] - 114 / 255) < 1e-6);   // grey padding in every plane
  assert.notDeepEqual(Array.from(chw.slice(0, 64)), Array.from(hwc.slice(0, 64)));   // the two layouts really differ
});

test('bundled model.json is consistent with what the decoder and preprocessor expect', () => {
  const m = JSON.parse(readFileSync(new URL('../assets/models/model.json', import.meta.url), 'utf8'));
  if (!m.installed) return;                                               // placeholder build: nothing to check
  assert.equal(m.imgsz, 640);
  assert.ok(['nchw', 'nhwc'].includes(m.input_layout) && ['channels_first', 'channels_last'].includes(m.output_layout));
  assert.equal(m.input_layout, 'nchw');                                   // verified against the real LiteRT file, see mobile/README.md
  assert.ok(m.labels.length >= 1 && m.labels.every((l: string) => typeof l === 'string'));
  assert.ok(readFileSync(new URL('../assets/models/model.tflite', import.meta.url)).byteLength > 1_000_000, 'model.tflite is a real file');
  assert.ok(m.warning, 'a weak-label development model must carry a visible warning');
});
