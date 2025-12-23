# embedding-classifier

Text embeddings with all-MiniLM-L6-v2 for semantic classification.

## Install

```bash
npm install
npm run build
```

## API

```typescript
import { embed, compare } from "./dist/index.js";

const embedding = await embed("some text");
const avgEmbedding = await embed(["text 1", "text 2", "text 3"]);
const distance = compare(embeddingA, embeddingB); // 0 = identical, 2 = opposite
```

## Building a Category Classifier

Create semantic classifiers by averaging embeddings of LLM-generated example phrases.

### 1. Generate category examples with an LLM

Prompt:

> Generate 20 short phrases matching "things related to pokemon, but not charizard" (this would be the input from the user).

### 2. Create the category embedding

```typescript
const categoryEmbedding = await embed([
  "Pikachu evolution",
  "catching wild pokemon",
  "gym badge collection",
  "pokedex completion",
  // ... more from LLM
]);
```

### 3. Classify new text

```typescript
const threshold = 0.5;

async function isInCategory(text: string, categoryEmb: number[]) {
  return compare(await embed(text), categoryEmb) < threshold;
}

await isInCategory("Squirtle uses water gun", categoryEmbedding); // true
await isInCategory("Charizard breathes fire", categoryEmbedding);  // false
await isInCategory("The stock market crashed", categoryEmbedding); // false
```

### Tips

- More phrases = better category representation
- Use negative examples as a separate embedding to filter against
- Lower threshold = stricter matching
