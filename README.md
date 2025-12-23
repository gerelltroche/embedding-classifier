# embedding-classifier

Text embeddings with all-MiniLM-L6-v2 for semantic classification. DISCLAIMER: I haven't tested this, but i've done similar before.

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

Use an LLM to break down a query into conditions, then check posts against all of them.

### 1. Parse user query with an LLM

User input: `"Find posts like football but not NFL"`

Prompt a small LLM (e.g. qwen3-4b) to extract conditions:

```typescript
type Condition = { query: string; isPositive: boolean };

// LLM output:
const conditions: Condition[] = [
  { query: "football", isPositive: true },
  { query: "NFL", isPositive: false },
];
```

### 2. Embed each condition

```typescript
const conditionEmbeddings = await Promise.all(
  conditions.map(async (c) => ({
    embedding: await embed(c.query),
    isPositive: c.isPositive,
  }))
);
```

### 3. Check if a post matches ALL conditions

```typescript
const threshold = 0.5;

async function matchesCategory(postText: string, conditions) {
  const postEmb = await embed(postText);

  for (const { embedding, isPositive } of conditions) {
    const distance = compare(postEmb, embedding);
    const isClose = distance < threshold;

    // Positive conditions: post should be close
    // Negative conditions: post should be far
    if (isPositive !== isClose) {
      return false;
    }
  }
  return true;
}

await matchesCategory("College football game highlights", conditionEmbeddings); // true
await matchesCategory("NFL draft picks announced", conditionEmbeddings);        // false
await matchesCategory("Stock market news", conditionEmbeddings);                // false
```

### Tips

- Post must match ALL conditions (close to positives, far from negatives)
- Adjust threshold based on your data
- Small models like qwen3-4b can handle the condition extraction
