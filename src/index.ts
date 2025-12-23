import { pipeline, type FeatureExtractionPipeline } from "@xenova/transformers";

const MODEL_NAME = "Xenova/all-MiniLM-L6-v2";

let extractor: FeatureExtractionPipeline | null = null;

async function getExtractor(): Promise<FeatureExtractionPipeline> {
  if (!extractor) {
    extractor = await pipeline("feature-extraction", MODEL_NAME);
  }
  return extractor;
}

export type Embedding = number[];

/**
 * Generate an embedding for the given text(s) using all-MiniLM-L6-v2.
 * If an array is provided, returns the averaged (and normalized) embedding.
 * Returns a 384-dimensional vector.
 */
export async function embed(text: string | string[]): Promise<Embedding> {
  const ext = await getExtractor();

  if (typeof text === "string") {
    const output = await ext(text, { pooling: "mean", normalize: true });
    return Array.from(output.data as Float32Array);
  }

  if (text.length === 0) {
    throw new Error("Cannot embed empty array");
  }

  const embeddings = await Promise.all(
    text.map(async (t) => {
      const output = await ext(t, { pooling: "mean", normalize: true });
      return Array.from(output.data as Float32Array);
    })
  );

  const dim = embeddings[0].length;
  const avg = new Array(dim).fill(0);

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      avg[i] += emb[i];
    }
  }

  let norm = 0;
  for (let i = 0; i < dim; i++) {
    avg[i] /= embeddings.length;
    norm += avg[i] * avg[i];
  }
  norm = Math.sqrt(norm);

  for (let i = 0; i < dim; i++) {
    avg[i] /= norm;
  }

  return avg;
}

/**
 * Calculate the cosine distance between two embeddings.
 * Returns a value between 0 (identical) and 2 (opposite).
 */
export function compare(a: Embedding, b: Embedding): number {
  if (a.length !== b.length) {
    throw new Error(`Embedding dimensions must match: ${a.length} vs ${b.length}`);
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const cosineSimilarity = dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  return 1 - cosineSimilarity;
}

// Demo usage
async function main() {
  console.log("Loading model...");

  const text1 = "The cat sat on the mat.";
  const text2 = "A feline rested on the rug.";
  const text3 = "The stock market crashed today.";

  console.log("\nGenerating embeddings...");
  const emb1 = await embed(text1);
  const emb2 = await embed(text2);
  const emb3 = await embed(text3);

  console.log(`\nEmbedding dimension: ${emb1.length}`);

  console.log("\nComparing texts:");
  console.log(`"${text1}" vs "${text2}"`);
  console.log(`  Distance: ${compare(emb1, emb2).toFixed(4)}`);

  console.log(`\n"${text1}" vs "${text3}"`);
  console.log(`  Distance: ${compare(emb1, emb3).toFixed(4)}`);

  console.log(`\n"${text2}" vs "${text3}"`);
  console.log(`  Distance: ${compare(emb2, emb3).toFixed(4)}`);
}

main().catch(console.error);
