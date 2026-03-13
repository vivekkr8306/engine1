import { db } from '../db/database';
import { getTextEmbedding, cosineSimilarity } from '../ai/modelManager';

export interface GalleryImage {
  id: string;
  uri: string;
  createdAt: number;
}

export const RetrievalFunction = async (
  query: string,
  allImages: GalleryImage[] = []
): Promise<GalleryImage[]> => {

  console.log('Starting hybrid search for:', query);

  const cleanQuery   = query.trim().toLowerCase();
  const finalResults: GalleryImage[] = [];
  const seenIds      = new Set<string>();

  try {
    // 1. Empty query → return all recent images
    if (!cleanQuery) {
      await db.transaction(async (tx) => {
        const result = await tx.execute(
          'SELECT id, path as uri, created_at as createdAt FROM documents ORDER BY created_at DESC LIMIT 100'
        );
        (result.rows || []).forEach((row: any) => {
          finalResults.push({
            id: row.id.toString(),
            uri: row.uri,
            createdAt: row.createdAt,
          });
        });
      });
      return finalResults;
    }

    // 2. Get text embedding via MobileCLIP (on-device, outside transaction)
    const textVector = await getTextEmbedding(cleanQuery);
    // Already normalized inside getTextEmbedding

    // 3. Search inside transaction
    await db.transaction(async (tx) => {

      // A. Lexical search — LIKE match on OCR text
      const textRes = await tx.execute(
        'SELECT id, path as uri, created_at as createdAt FROM documents WHERE ocr_text LIKE ?',
        [`%${cleanQuery}%`]
      );
      for (const match of (textRes.rows || [])) {
        const idStr = match.id?.toString();
        if (idStr && !seenIds.has(idStr)) {
          seenIds.add(idStr);
          finalResults.push({
            id: idStr,
            uri: match.uri as string,
            createdAt: match.createdAt as number,
          });
        }
      }

      // B. Semantic vector search — cosine similarity in JS
      const vectorRes = await tx.execute(
        'SELECT id, path as uri, created_at as createdAt, embedding FROM documents WHERE embedding IS NOT NULL'
      );

      const vectorMatches: {
        id: string; uri: string; createdAt: number; score: number;
      }[] = [];

      for (const doc of (vectorRes.rows || [])) {
        const idStr = doc.id?.toString();
        if (idStr && seenIds.has(idStr)) continue; // already in lexical results
        if (!doc.embedding) continue;

        const docVec     = new Float32Array(doc.embedding as ArrayBuffer);
        const similarity = cosineSimilarity(textVector, docVec);

        vectorMatches.push({
          id: idStr!,
          uri: doc.uri as string,
          createdAt: doc.createdAt as number,
          score: similarity,
        });
      }

      // Sort by similarity, take top 50
      vectorMatches.sort((a, b) => b.score - a.score);

      const TOP_K = 50;
      for (let i = 0; i < Math.min(TOP_K, vectorMatches.length); i++) {
        const match = vectorMatches[i];
        if (!seenIds.has(match.id)) {
          seenIds.add(match.id);
          finalResults.push({
            id: match.id,
            uri: match.uri,
            createdAt: match.createdAt,
          });
        }
      }
    });

    console.log(`✅ Search complete — ${finalResults.length} results`);
    return finalResults;

  } catch (error) {
    console.error('❌ Search failed:', error);

    // Fallback: lexical only (works even when model fails)
    try {
      const fallback: GalleryImage[] = [];
      await db.transaction(async (tx) => {
        const res = await tx.execute(
          'SELECT id, path as uri, created_at as createdAt FROM documents WHERE ocr_text LIKE ? ORDER BY created_at DESC LIMIT 50',
          [`%${cleanQuery}%`]
        );
        (res.rows || []).forEach((row: any) => {
          fallback.push({
            id: row.id?.toString() || '',
            uri: row.uri as string,
            createdAt: row.createdAt as number,
          });
        });
      });
      console.log(`⚠️ Fallback lexical search: ${fallback.length} results`);
      return fallback;

    } catch (fallbackError) {
      console.error('❌ Fallback also failed:', fallbackError);
      return allImages;
    }
  }
};