// On-device inference. Everything here runs locally: no network calls anywhere.
//   photo -> resize (native) -> JPEG decode (JS) -> letterbox + normalise (pure TS) -> TFLite -> decode + NMS (pure TS)
import { ImageManipulator, SaveFormat } from 'expo-image-manipulator';
import { loadTensorflowModel, type TensorflowModel } from 'react-native-fast-tflite';
import jpeg from 'jpeg-js';
import { base64ToBytes, rgbaToTensor, type InputLayout } from '../core/preprocess';
import { decodeYolo, toOriginal, type Detection, type Layout } from '../core/yolo';
import modelInfo from '../../assets/models/model.json';

export type ModelInfo = {
  installed: boolean; version?: string; imgsz: number; labels: string[]; int8: boolean; format: string;
  output_layout: Layout; input_layout: InputLayout; input_dtype: 'float32' | 'uint8'; warning?: string;
};
export const MODEL: ModelInfo = modelInfo as ModelInfo;

export type Result = { detections: Detection[]; width: number; height: number; ms: number };

let modelPromise: Promise<TensorflowModel> | null = null;

/** Loads the bundled model once. Rejects with a readable message if no trained model has been installed yet. */
export function loadModel(): Promise<TensorflowModel> {
  if (!MODEL.installed) return Promise.reject(new Error('No trained model is installed in this build yet (see mobile/README.md).'));
  // core-ml on iOS and the GPU delegate on Android are used when available (see app.json plugin options); CPU otherwise.
  return (modelPromise ??= loadTensorflowModel(require('../../assets/models/model.tflite'), 'default'));
}

/** `srcW`/`srcH` come from the image picker's asset, so no extra decode is needed to read the orientation. */
export async function detect(uri: string, srcW: number, srcH: number, opts: { conf?: number; iou?: number } = {}): Promise<Result> {
  const t0 = Date.now();
  const model = await loadModel();
  const size = MODEL.imgsz;

  // Downscale natively so the long side equals the model size: cheap, and keeps JS decoding small.
  const small = await (await ImageManipulator.manipulate(uri).resize(srcH > srcW ? { height: size } : { width: size }).renderAsync())
    .saveAsync({ format: SaveFormat.JPEG, compress: 0.92, base64: true });

  const img = jpeg.decode(base64ToBytes(small.base64 ?? ''), { useTArray: true, formatAsRGBA: true });
  const { input, letterbox } = rgbaToTensor(img.data, img.width, img.height, size, MODEL.input_layout ?? 'nhwc');

  const feed = MODEL.input_dtype === 'uint8' ? Uint8Array.from(input, (v) => Math.round(v * 255)) : input;
  const outputs = await model.run([feed.buffer as ArrayBuffer]);
  const raw = new Float32Array(outputs[0]);

  const dets = decodeYolo(raw, {
    numClasses: MODEL.labels.length, imgsz: size, layout: MODEL.output_layout,
    confThreshold: opts.conf ?? 0.25, iouThreshold: opts.iou ?? 0.45,
  });
  return { detections: toOriginal(dets, letterbox, img.width, img.height, MODEL.labels), width: img.width, height: img.height, ms: Date.now() - t0 };
}
