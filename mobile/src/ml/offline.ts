// The on-device YOLO11n path, adapted to the core's OnDevice signature.
import { fromDevice, type OnDevice } from '../core/scan';
import { MODEL, detect } from './detector';

export const onDevice: OnDevice = async (photo) => {
  const r = await detect(photo.uri, photo.width, photo.height);          // throws a readable error if no model is bundled
  return { detections: fromDevice(r.detections), modelVersion: MODEL.version ?? 'on-device', imageSize: { width: r.width, height: r.height } };
};
