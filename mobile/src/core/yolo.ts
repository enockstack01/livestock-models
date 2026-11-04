// Pure TypeScript (no React Native imports) so it can be unit-tested with plain Node:  npm test
// YOLO11 detection post-processing: layout detection, box decoding, class-aware NMS, un-letterboxing.

export type Box = { x1: number; y1: number; x2: number; y2: number };
export type Detection = { classId: number; label: string; score: number; box: Box };
export type Layout = 'channels_first' | 'channels_last'; // [1, 4+nc, N]  vs  [1, N, 4+nc]

export type Letterbox = { scale: number; padX: number; padY: number; newW: number; newH: number; size: number };

/** Scale an image to fit size x size keeping aspect ratio, centred, the rest padded (same as ultralytics). */
export function letterboxParams(srcW: number, srcH: number, size: number): Letterbox {
  const scale = Math.min(size / srcW, size / srcH);
  const newW = Math.round(srcW * scale);
  const newH = Math.round(srcH * scale);
  return { scale, padX: Math.floor((size - newW) / 2), padY: Math.floor((size - newH) / 2), newW, newH, size };
}

export type DecodeOptions = {
  numClasses: number;
  imgsz: number;
  layout: Layout;
  confThreshold?: number; // low default (0.25) favours recall: a missed infectious disease costs more than a false alarm
  iouThreshold?: number;
  maxDet?: number;
};

/** Boxes from the raw network output, in model-input pixel coordinates (cx,cy,w,h -> x1,y1,x2,y2). */
export function decodeYolo(out: ArrayLike<number>, o: DecodeOptions): { classId: number; score: number; box: Box }[] {
  const C = 4 + o.numClasses;
  if (out.length % C !== 0) throw new Error(`output length ${out.length} is not a multiple of 4+nc=${C}; wrong class count or a segmentation model?`);
  const N = out.length / C;
  const at = (anchor: number, ch: number) => (o.layout === 'channels_first' ? out[ch * N + anchor] : out[anchor * C + ch]);
  const conf = o.confThreshold ?? 0.25;

  const cand: { classId: number; score: number; cx: number; cy: number; w: number; h: number }[] = [];
  for (let a = 0; a < N; a++) {
    let best = -1;
    let bestScore = 0;
    for (let c = 0; c < o.numClasses; c++) {
      const s = at(a, 4 + c);
      if (s > bestScore) { bestScore = s; best = c; }
    }
    if (best >= 0 && bestScore >= conf) cand.push({ classId: best, score: bestScore, cx: at(a, 0), cy: at(a, 1), w: at(a, 2), h: at(a, 3) });
  }
  // TFLite/CoreML exports emit box coordinates normalised to 0..1, ONNX emits pixels. Detect from the data.
  const maxCoord = cand.reduce((m, d) => Math.max(m, d.cx, d.cy, d.w, d.h), 0);
  const k = cand.length > 0 && maxCoord <= 2 ? o.imgsz : 1;
  const boxes = cand.map((d) => ({
    classId: d.classId,
    score: d.score,
    box: { x1: (d.cx - d.w / 2) * k, y1: (d.cy - d.h / 2) * k, x2: (d.cx + d.w / 2) * k, y2: (d.cy + d.h / 2) * k },
  }));
  return nms(boxes, o.iouThreshold ?? 0.45, o.maxDet ?? 50);
}

export function iou(a: Box, b: Box): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union <= 0 ? 0 : inter / union;
}

/** Greedy non-maximum suppression, per class (a lesion box must not suppress an overlapping animal box). */
export function nms<T extends { classId: number; score: number; box: Box }>(items: T[], iouThr: number, maxDet: number): T[] {
  const sorted = [...items].sort((a, b) => b.score - a.score);
  const keep: T[] = [];
  for (const it of sorted) {
    if (keep.length >= maxDet) break;
    if (keep.every((k) => k.classId !== it.classId || iou(k.box, it.box) < iouThr)) keep.push(it);
  }
  return keep;
}

/** Map boxes from letterboxed model space back to the original image's pixels, clipped to the image. */
export function toOriginal(dets: { classId: number; score: number; box: Box }[], lb: Letterbox, srcW: number, srcH: number, labels: string[]): Detection[] {
  const clip = (v: number, max: number) => Math.max(0, Math.min(max, v));
  return dets
    .map((d) => ({
      classId: d.classId,
      label: labels[d.classId] ?? `class_${d.classId}`,
      score: d.score,
      box: {
        x1: clip((d.box.x1 - lb.padX) / lb.scale, srcW),
        y1: clip((d.box.y1 - lb.padY) / lb.scale, srcH),
        x2: clip((d.box.x2 - lb.padX) / lb.scale, srcW),
        y2: clip((d.box.y2 - lb.padY) / lb.scale, srcH),
      },
    }))
    .filter((d) => d.box.x2 - d.box.x1 > 1 && d.box.y2 - d.box.y1 > 1);
}
