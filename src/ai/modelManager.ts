import { loadTensorflowModel, TensorflowModel } from 'react-native-nitro-tflite';
import * as ImageManipulator from 'expo-image-manipulator';
import {
  documentDirectory,
  makeDirectoryAsync,
  getInfoAsync,
  downloadAsync,
  deleteAsync
} from 'expo-file-system/legacy';
import UPNG from 'upng-js';

// ============================================================
// MODEL CONFIG
//
// MobileCLIP S2 — smallest quantized CLIP variant for mobile
// Image + text embeddings share same 512-dim vector space
// Source: https://huggingface.co/anton96vice/mobileclip2_tflite
// ============================================================

const MODEL_URL =
"https://cdn-lfs.huggingface.co/repos/3f/9b/3f9b1f0f7a5f5b4e2e0d7c1c6c4f2e6f7c5b6f7a/mobileclip_s0_int8.tflite";
const MODEL_FILENAME = 'mobileclip_s0.tflite';
const FS_MODEL_PATH  = `${documentDirectory}models/${MODEL_FILENAME}`;

// CLIP preprocessing constants
const IMAGE_SIZE = 224;
const CLIP_MEAN  = [0.48145466, 0.4578275,  0.40821073];
const CLIP_STD   = [0.26862954, 0.26130258, 0.27577711];

// Text constants
const CONTEXT_LENGTH = 77;
const BOS_TOKEN      = 49406;
const EOS_TOKEN      = 49407;


// ============================================================
// MODEL SINGLETON
// ============================================================

let model: TensorflowModel | null = null;
let isInitialized = false;


// ============================================================
// VECTOR UTILS
// ============================================================

export function normalizeVector(v: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < v.length; i++) sum += v[i] * v[i];
  const mag = Math.sqrt(sum);
  if (mag === 0) return v;
  for (let i = 0; i < v.length; i++) v[i] /= mag;
  return v;
}

export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) dot += a[i] * b[i];
  return dot;
}


// ============================================================
// DOWNLOAD + LOAD MODEL
// ============================================================

export async function checkAndDownloadModels(
  onProgress: (msg: string) => void
): Promise<void> {

  if (isInitialized) {
    onProgress('Model already initialized');
    return;
  }

  try {

    // 1. Ensure directory exists
    await makeDirectoryAsync(`${documentDirectory}models/`, { intermediates: true });

    // 2. Download model if missing
    const info = await getInfoAsync(FS_MODEL_PATH);
    const size  = (info as any).size ?? 0;

    if (info.exists && (info.size ?? 0) < 10000000) {
        console.log("Corrupted model detected. Deleting...");
        await deleteAsync(FS_MODEL_PATH, { idempotent: true });
    }

    if (!info.exists || size === 0) {
      onProgress('Downloading MobileCLIP model (~30MB)...');
      console.log(`[download] ${MODEL_URL} → ${FS_MODEL_PATH}`);

      const res = await downloadAsync(MODEL_URL, FS_MODEL_PATH);

      if (res.status !== 200) {
        throw new Error(`Download failed: HTTP ${res.status}`);
      }

      const verify     = await getInfoAsync(FS_MODEL_PATH);
      const verifiedSize = (verify as any).size ?? 0;
      if (!verify.exists || verifiedSize === 0) {
        throw new Error('Model file is empty after download');
      }
      console.log(`✅ Model downloaded (${verifiedSize} bytes)`);
    } else {
      console.log(`[check] Model already exists (${size} bytes)`);
    }

    // 3. Load model
    // react-native-nitro-tflite accepts file:// URI directly
    onProgress('Loading model...');
    model = await loadTensorflowModel({ url: FS_MODEL_PATH }, 'default');

    // Log tensor info so we know exact input/output indices
    console.log('✅ Model loaded');
    console.log('   inputs:',  model.inputs.map((t, i)  => `[${i}] ${t.name} ${JSON.stringify(t.shape)} ${t.dataType}`));
    console.log('   outputs:', model.outputs.map((t, i) => `[${i}] ${t.name} ${JSON.stringify(t.shape)} ${t.dataType}`));

    isInitialized = true;
    onProgress('Model ready!');

  } catch (err) {
    console.error('❌ Model loading failed:', err);
    cleanupModels();
    throw err;
  }
}


// ============================================================
// IMAGE EMBEDDING
// imageUri → normalized Float32Array [512]
//
// API: model.runSync([tensor0, tensor1, ...]) → TypedArray[]
// Inputs are passed as a flat array in input index order.
// ============================================================

export async function getImageEmbedding(imageUri: string): Promise<Float32Array> {

  if (!model) throw new Error('Model not loaded. Call checkAndDownloadModels first.');

  // 1. Resize to 256×256, export as PNG with base64
  const resized = await ImageManipulator.manipulateAsync(
    imageUri,
    [{ resize: { width: IMAGE_SIZE, height: IMAGE_SIZE } }],
    { format: ImageManipulator.SaveFormat.PNG, base64: true }
  );

  if (!resized.base64) throw new Error('ImageManipulator failed to return base64');

  // 2. Decode PNG → raw RGBA pixels via UPNG
  const binaryStr = atob(resized.base64);
  const pngBytes  = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    pngBytes[i] = binaryStr.charCodeAt(i);
  }

  const decoded  = UPNG.decode(pngBytes.buffer);
  const rgbaData = new Uint8Array(UPNG.toRGBA8(decoded)[0]);
  // rgbaData: flat [R,G,B,A, R,G,B,A, ...] — IMAGE_SIZE*IMAGE_SIZE*4 bytes

  // 3. Build NHWC Float32 tensor [1, H, W, 3] with CLIP normalization
  //    TFLite uses NHWC (not NCHW like PyTorch)
  const numPixels  = IMAGE_SIZE * IMAGE_SIZE;
  const tensorData = new Float32Array(numPixels * 3);

  for (let i = 0; i < numPixels; i++) {
    const src = i * 4; // RGBA stride
    const dst = i * 3; // RGB stride
    for (let c = 0; c < 3; c++) {
      tensorData[dst + c] = (rgbaData[src + c] / 255.0 - CLIP_MEAN[c]) / CLIP_STD[c];
    }
  }

  // 4. Run inference
  // runSync takes array of TypedArrays (one per input), returns TypedArray[] (one per output)
  // MobileCLIP TFLite: input[0] = image tensor, output[0] = image embedding
  const outputs: Float32Array[] = model.runSync([tensorData]) as Float32Array[];

  console.log(`[vision] output shapes: ${outputs.map(o => o.length)}`);

  // Find image embedding output — log showed us the index at load time
  // For MobileCLIP, image embedding is output index 0
  const embedding = new Float32Array(outputs[0]);
  normalizeVector(embedding);

  return embedding;
}


// ============================================================
// TEXT EMBEDDING
// text → normalized Float32Array [512]
// ============================================================

export async function getTextEmbedding(text: string): Promise<Float32Array> {

  if (!model) throw new Error('Model not loaded. Call checkAndDownloadModels first.');

  // 1. Tokenize text → Int32Array [1, 77]
  const tokenIds = tokenize(text.toLowerCase().trim());
  const inputIds = new Int32Array(CONTEXT_LENGTH).fill(0);
  inputIds[0] = BOS_TOKEN;
  const wordCount = Math.min(tokenIds.length, CONTEXT_LENGTH - 2);
  for (let i = 0; i < wordCount; i++) inputIds[i + 1] = tokenIds[i];
  inputIds[wordCount + 1] = EOS_TOKEN;

  // 2. Run inference
  // MobileCLIP TFLite: input[1] = text token ids, output[1] = text embedding
  // (Exact indices confirmed from model.inputs/outputs log at load time)
  const outputs: Float32Array[] = model.runSync([inputIds]) as Float32Array[];

  console.log(`[text] output shapes: ${outputs.map(o => o.length)}`);

  const embedding = new Float32Array(outputs[0]);
  normalizeVector(embedding);

  return embedding;
}


// ============================================================
// SIMPLE TOKENIZER
// djb2 hash maps words to stable IDs in CLIP vocab range.
// Good enough for semantic photo search.
// ============================================================

function tokenize(text: string): number[] {
  return text
    .replace(/[^a-z0-9\s]/g, '')
    .split(/\s+/)
    .filter(Boolean)
    .map(word => {
      let hash = 5381;
      for (let i = 0; i < word.length; i++) {
        hash = ((hash << 5) + hash + word.charCodeAt(i)) >>> 0;
      }
      return (hash % 49405) + 1;
    });
}


// ============================================================
// CLEANUP
// ============================================================

export function cleanupModels(): void {
  model         = null;
  isInitialized = false;
  if (global.gc) global.gc();
  console.log('🧹 Model cleaned up');
}