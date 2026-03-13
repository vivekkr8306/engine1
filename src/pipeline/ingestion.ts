import TextRecognition from '@react-native-ml-kit/text-recognition';
import { getImageEmbedding } from '../ai/modelManager';
import { db } from '../db/database';

export interface GalleryImage {
  id: string;
  uri: string;
  createdAt: number;
}

export const IngestionFunction = async (imageUri: string): Promise<GalleryImage> => {
  console.log('Starting ingestion for:', imageUri);
  const createdAt = Date.now();

  try {
    // 1. OCR via ML Kit (on-device, no internet)
    const ocrResult = await TextRecognition.recognize(imageUri);
    const cleanText = ocrResult.text.replace(/\n/g, ' ').trim().toLowerCase();
    console.log(`OCR: ${cleanText.length > 0 ? `"${cleanText.slice(0, 80)}..."` : 'none'}`);

    // 2. Visual embedding via MobileCLIP TFLite (on-device)
    const imageVector = await getImageEmbedding(imageUri);
    // Already normalized inside getImageEmbedding

    // 3. Store in SQLite as BLOB
    const vectorBlob = imageVector.buffer as ArrayBuffer;

    let rowId = 0;
    await db.transaction(async (tx) => {
      const docResult = tx.execute(
        'INSERT INTO documents (path, ocr_text, embedding, created_at) VALUES (?, ?, ?, ?)',
        [imageUri, cleanText, vectorBlob, createdAt]
      );
      rowId = (await docResult).insertId!;
    });

    console.log(`✅ Ingested and stored (id=${rowId})`);
    return { id: rowId.toString(), uri: imageUri, createdAt };

  } catch (error) {
    console.error(`❌ Ingestion failed for ${imageUri}:`, error);

    // Fallback: store with OCR text only, no embedding
    try {
      const ocrResult = await TextRecognition.recognize(imageUri);
      const cleanText = ocrResult.text.replace(/\n/g, ' ').trim().toLowerCase();

      let rowId = 0;
      await db.transaction(async (tx) => {
        const docResult = tx.execute(
          'INSERT INTO documents (path, ocr_text, embedding, created_at) VALUES (?, ?, ?, ?)',
          [imageUri, cleanText, null, createdAt]
        );
        rowId = (await docResult).insertId!;
      });

      console.log('⚠️ Stored with OCR only (no embedding)');
      return { id: rowId.toString(), uri: imageUri, createdAt };

    } catch (fallbackError) {
      console.error('❌ Fallback also failed:', fallbackError);
      throw error;
    }
  }
};