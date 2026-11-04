// Pure TypeScript image preprocessing (runs on the phone's CPU, no network, no native code).
import { letterboxParams, type Letterbox } from './yolo.ts';

const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';

/** base64 -> bytes without relying on atob/Buffer being present in the JS runtime. */
export function base64ToBytes(b64: string): Uint8Array {
  const clean = b64.replace(/^data:[^,]*,/, '').replace(/[^A-Za-z0-9+/]/g, '');
  const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
  let o = 0;
  for (let i = 0; i < clean.length; i += 4) {
    const n = [0, 1, 2, 3].map((j) => (i + j < clean.length ? B64.indexOf(clean[i + j]) : -1));
    const v = ((n[0] & 63) << 18) | ((n[1] & 63) << 12) | (((n[2] < 0 ? 0 : n[2]) & 63) << 6) | ((n[3] < 0 ? 0 : n[3]) & 63);
    if (o < out.length) out[o++] = (v >> 16) & 255;
    if (n[2] >= 0 && o < out.length) out[o++] = (v >> 8) & 255;
    if (n[3] >= 0 && o < out.length) out[o++] = v & 255;
  }
  return out;
}

export type Prepared = { input: Float32Array; letterbox: Letterbox; srcW: number; srcH: number };

/**
 * RGBA pixels (e.g. from jpeg-js) -> letterboxed, normalised float32 tensor [3*size*size] in 0..1, padded with grey 114 like
 * ultralytics. Bilinear resize. `layout` must match the model: ultralytics' LiteRT/TFLite export wants 'nchw' (planar R,G,B),
 * NOT the 'nhwc' (interleaved) that most TFLite models use. Feeding the wrong one gives plausible-looking garbage, with no error.
 */
export type InputLayout = 'nhwc' | 'nchw';

export function rgbaToTensor(rgba: ArrayLike<number>, srcW: number, srcH: number, size: number, layout: InputLayout = 'nhwc'): Prepared {
  const lb = letterboxParams(srcW, srcH, size);
  const input = new Float32Array(size * size * 3).fill(114 / 255);
  const sx = srcW / lb.newW;
  const sy = srcH / lb.newH;
  for (let y = 0; y < lb.newH; y++) {
    const fy = Math.max(0, Math.min(srcH - 1, (y + 0.5) * sy - 0.5));
    const y0 = Math.floor(fy);
    const y1 = Math.min(srcH - 1, y0 + 1);
    const wy = fy - y0;
    for (let x = 0; x < lb.newW; x++) {
      const fx = Math.max(0, Math.min(srcW - 1, (x + 0.5) * sx - 0.5));
      const x0 = Math.floor(fx);
      const x1 = Math.min(srcW - 1, x0 + 1);
      const wx = fx - x0;
      const pos = (y + lb.padY) * size + (x + lb.padX);
      for (let c = 0; c < 3; c++) {
        const p00 = rgba[(y0 * srcW + x0) * 4 + c];
        const p10 = rgba[(y0 * srcW + x1) * 4 + c];
        const p01 = rgba[(y1 * srcW + x0) * 4 + c];
        const p11 = rgba[(y1 * srcW + x1) * 4 + c];
        const top = p00 + (p10 - p00) * wx;
        const bot = p01 + (p11 - p01) * wx;
        input[layout === 'nchw' ? c * size * size + pos : pos * 3 + c] = (top + (bot - top) * wy) / 255;
      }
    }
  }
  return { input, letterbox: lb, srcW, srcH };
}
