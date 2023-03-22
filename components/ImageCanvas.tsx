import { useRef, useState } from 'react';
import { IMAGE_URLS } from '../data/sample-image-urls';
import { inferenceSqueezenet } from '../utils/predict';
import styles from '../styles/Home.module.css';
import * as ort from 'onnxruntime-web';
import { getSqueezenetSession } from '../utils/modelHelper';
interface Props {
  height: number;
  width: number;
}

const ImageCanvas = (props: Props) => {

  const canvasRefA = useRef<HTMLCanvasElement>(null);
	const canvasRefB = useRef<HTMLCanvasElement>(null);

  var image: HTMLImageElement;
  const [topResultLabelA, setLabelA] = useState("");
  const [topResultConfidenceA, setConfidenceA] = useState("");
  const [inferenceTimeA, setInferenceTimeA] = useState("");

	const [topResultLabelB, setLabelB] = useState("");
  const [topResultConfidenceB, setConfidenceB] = useState("");
  const [inferenceTimeB, setInferenceTimeB] = useState("");

  // Load the image from the IMAGE_URLS array
  const getImage = () => {
    var sampleImageUrls: Array<{ text: string; value: string }> = IMAGE_URLS;
    var random = Math.floor(Math.random() * (9 - 0 + 1) + 0);
    return sampleImageUrls[random];
  }

  // Draw image and other  UI elements then run inference
  const displayImageAndRunInference = async (sessionType: string,
		setLabelFcn: (x: string) => void,
		setConfidenceFcn: (x: string) => void,
		setInferenceTimeFcn: (x: string) => void) => {
		const session = await getSqueezenetSession(sessionType);
		if (!session) {
			setLabelFcn("Session load failure")
			return;
		}
    // Get the image
    image = new Image();
    var sampleImage = getImage();
    image.src = sampleImage.value;

    // Clear out previous values.
    setLabelFcn(`Inferencing...`);
    setConfidenceFcn("");
    setInferenceTimeFcn("");

    // Draw the image on the canvas
    const canvas = sessionType === "wasm" ? canvasRefA.current : canvasRefB.current;
		console.log("Canvas", canvas);
    const ctx = canvas!.getContext('2d');
    image.onload = () => {
      ctx!.drawImage(image, 0, 0, props.width, props.height);
    }

    // Run the inference
    submitInference(session, setLabelFcn, setConfidenceFcn, setInferenceTimeFcn);
  };

	const displayAndRunWasmInference = async () => displayImageAndRunInference("wasm",
		setLabelA, setConfidenceA, setInferenceTimeA);
	const displayAndRunWebGLInference = async () => displayImageAndRunInference("webgl",
		setLabelB, setConfidenceB, setInferenceTimeB);
  const submitInference = async (session: ort.InferenceSession,
		setLabelFcn: (x: string) => void,
		setConfidenceFcn: (x: string) => void,
		setInferenceTimeFcn: (x: string) => void) => {
		var totalTime = 0;
		const numReps = 100;
		for (let i = 0; i < numReps; i++) {
			// Get the image data from the canvas and submit inference.
			var result = await inferenceSqueezenet(image.src, session);
			if (result) {
				var [inferenceResult,singleTime] = result
				totalTime += singleTime;
				console.log(singleTime);
				// Get the highest confidence.
				var topResult = inferenceResult[0];
				setLabelFcn(`${i}/100 completed`);
			} else {
				i--;
			}
		}
    // Update the label and confidence
    setLabelFcn(topResult.name.toUpperCase());
    setConfidenceFcn(topResult.probability);
    setInferenceTimeFcn(`Inference speed avg over ${numReps} trials: ${1000* totalTime / numReps} ms`);

  };

  return (
    <>
      <button
        className={styles.grid}
        onClick={displayAndRunWasmInference} >
        Run Squeezenet WASM inference
      </button>
      <br/>
      <canvas ref={canvasRefA} width={props.width} height={props.height} />
      <span>{topResultLabelA} {topResultConfidenceA}</span>
      <span>{inferenceTimeA}</span>
			<br />
			<button
        className={styles.grid}
        onClick={displayAndRunWebGLInference} >
        Run Squeezenet WebGL inference
      </button>
      <br/>
      <canvas ref={canvasRefB} width={props.width} height={props.height} />
      <span>{topResultLabelB} {topResultConfidenceB}</span>
      <span>{inferenceTimeB}</span>
    </>
  )
};

export default ImageCanvas;
