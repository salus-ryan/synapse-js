# Synapse.js

**An AI-Native Reactive Framework for JavaScript/TypeScript**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Node: >=18](https://img.shields.io/badge/Node-%3E%3D18-green.svg)]()

Synapse.js treats AI inference as a **reactive primitive**. Like signals are to Solid.js, *Synapses* are to AI: composable, streamable, and fine-grained reactive nodes in a computation graph.

```typescript
import { createSignal, createSynapse, createEffect, setDefaultProvider, createOpenAIProvider } from 'synapse-js';

setDefaultProvider(createOpenAIProvider());

const [topic, setTopic] = createSignal('quantum computing');

const explainer = createSynapse({
  model: 'gpt-4o-mini',
  signature: 'topic -> explanation, fun_fact',
  dependencies: () => ({ topic: topic() }),
});

createEffect(() => {
  const result = explainer.output();
  if (result) console.log(result.explanation);
});

// Changing the signal automatically triggers a new LLM call!
setTopic('black holes');
```

## Why Synapse.js?

| Feature | Vercel AI SDK | LangChain.js | **Synapse.js** |
|---------|:---:|:---:|:---:|
| Fine-grained reactivity | ✗ | ✗ | ✓ |
| DSPy-style signatures | ✗ | ✗ | ✓ |
| Multi-model braiding | ✗ | ✗ | ✓ |
| Reactive memory graph | ✗ | partial | ✓ |
| Streaming as signals | partial | ✗ | ✓ |
| 8-dot Braille encoding | ✗ | ✗ | ✓ |
| Zero dependencies | ✗ | ✗ | ✓ |

## Installation

```bash
npm install synapse-js
```

## Primitives

### `createSynapse` — Reactive LLM Inference
Wraps an LLM call in a reactive node with auto-dependency tracking, debouncing, cancellation, and streaming.

### `createBraid` — Multi-Model Composition
Compose multiple Synapses with strategies: `race` (fastest wins), `consensus` (majority agreement), `fallback` (try in order), `judge` (meta-model picks best).

### `createMemory` — Reactive Context Graph
Token-budgeted memory that auto-windows/summarizes. Reactive: downstream Synapses re-fire when context changes.

### `createAgent` — Autonomous Loop
Synapse + Memory + Tools in an autonomous reasoning loop.

### `createPipeline` — Declarative Chains
Output of one Synapse feeds the next. Observable intermediate results.

### `createTool` — Reactive Function Calling
Tools with reactive state: `lastResult`, `loading`, `callCount` are all signals.

### 8-Dot Braille Encoding
AI state encoded as Unicode Braille (U+2800–U+28FF) — simultaneously visual, tactile, and semantic. See [Architecture: Accessible-Native](#accessible-native-braille-encoding).

## Quick Start with Ollama (No API Keys)

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small model
ollama pull qwen2.5:0.5b

# Clone and run
git clone https://github.com/salus-ryan/synapse-js.git
cd synapse-js
npm install
npm run demo
```

## Multi-LLM Braid Demo

```bash
# With multiple local models:
ollama pull qwen2.5:0.5b
ollama pull llama3.2:1b
ollama pull gemma2:2b

npm run demo:braid
```

This races multiple models against each other, showing race/consensus/judge strategies with live Braille progress bars.

## Accessible-Native: Braille Encoding

Synapse.js encodes AI state as 8-dot Braille characters (256 unique patterns per character):

```typescript
import { confidenceBar, accessibleConfidence, createBrailleSignal } from 'synapse-js';

// Confidence as a tactile/visual bar
confidenceBar(0.75, 8); // → "⣿⣿⣿⣿⣿⣿⡇⠀"

// Accessible annotation (Braille + ARIA label)
accessibleConfidence(0.95, 'GPT-4o');
// → { braille: "⣿⣿⣿⣿⣿⣿⣿⡇", ariaLabel: "GPT-4o: 95%" }

// Reactive Braille signal from AI state
const brailleState = createBrailleSignal(synapse.state);
createEffect(() => {
  const { braille, ariaLabel } = brailleState();
  element.textContent = braille;        // Visual + tactile
  element.ariaLabel = ariaLabel;         // Screen reader
});
```

**Why?** The same encoding is simultaneously:
- **Visual** — dot patterns create data visualization for sighted users
- **Tactile** — Braille displays render them physically for blind users
- **Semantic** — screen readers announce meaningful labels
- **Compact** — 256 symbols = information-dense encoding (1 byte per char)

```bash
npm run demo:braille
```

## Project Structure

```
synapse-js/
├── src/
│   ├── index.ts                  # Public API
│   ├── core/
│   │   └── reactive.ts           # Fine-grained reactive engine
│   ├── primitives/
│   │   ├── synapse.ts            # Reactive LLM inference
│   │   ├── braid.ts             # Multi-model composition
│   │   ├── memory.ts            # Reactive context graph
│   │   ├── pipeline.ts          # Declarative chains
│   │   ├── tool.ts              # Reactive function calling
│   │   └── agent.ts             # Autonomous loop
│   ├── runtime/
│   │   └── providers.ts         # OpenAI-compatible provider
│   └── encoding/
│       └── braille.ts           # 8-dot Braille encoding
├── examples/
│   ├── demo-ollama.ts           # Full framework demo (local)
│   ├── demo-braid-multi-llm.ts  # Multi-model braid showcase
│   ├── demo-braille.ts          # Accessible encoding demo
│   ├── basic.ts                 # Minimal example
│   └── agent.ts                 # Agent with tools
├── tests/
│   └── run-tests.ts             # Test suite
├── package.json
├── tsconfig.json
└── tsup.config.ts
```

## Architecture

The reactive engine is **zero-dependency** and optimized for async/streaming operations:

- **State Nodes (Signals)**: Hold raw values (user input, config)
- **Compute Nodes (Synapses)**: Trigger LLM calls when deps change; manage cancellation/debounce
- **Effect Nodes**: Update UI or trigger side effects
- **Braid Nodes**: Compose multiple compute nodes with resolution strategies

## License

MIT
