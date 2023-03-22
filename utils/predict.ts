// Language: typescript
// Path: react-next\utils\predict.ts
import { getImageTensorFromPath } from './imageHelper';
import { runSqueezenetModel } from './modelHelper';
import * as ort from 'onnxruntime-web';

export async function inferenceSqueezenet(path: string, session: ort.InferenceSession): Promise<[any,number] | undefined> {
  // 1. Convert image to tensor
  const imageTensor = await getImageTensorFromPath(path);
  // 2. Run model

  const result = await runSqueezenetModel(imageTensor, session);
	if (result) {
		// 3. Return predictions and the amount of time it took to inference.
		const [predictions, inferenceTime] = result
		return [predictions, inferenceTime];
	}
}
