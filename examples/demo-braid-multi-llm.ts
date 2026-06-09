/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║     Synapse.js — Multi-LLM Braid Demo                          ║
 * ║                                                                  ║
 * ║  Demonstrates braiding responses from multiple LLMs             ║
 * ║  using race, consensus, and judge strategies.                   ║
 * ║                                                                  ║
 * ║  Uses Ollama with multiple models locally.                      ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * Prerequisites:
 *   1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
 *   2. Pull models:
 *        ollama pull qwen2.5:0.5b
 *        ollama pull llama3.2:1b
 *        ollama pull gemma2:2b
 *   3. Run: npx tsx examples/demo-braid-multi-llm.ts
 *
 * Or use OpenAI-compatible APIs by setting environment variables:
 *   OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
 */

import {
  createSignal,
  createEffect,
  createSynapse,
  createBraid,
  setDefaultProvider,
  createOpenAIProvider,
  confidenceBar,
  accessibleConfidence,
  semanticFingerprint,
  fingerprintSimilarity,
} from '../src/index';

// ─────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────

const USE_OLLAMA = !process.env.OPENAI_API_KEY;

const OLLAMA_MODELS = ['qwen2.5:0.5b', 'llama3.2:1b', 'gemma2:2b'];
const OPENAI_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-3.5-turbo'];

const models = USE_OLLAMA ? OLLAMA_MODELS : OPENAI_MODELS;

const provider = createOpenAIProvider(
  USE_OLLAMA
    ? { apiKey: 'ollama', baseURL: 'http://localhost:11434/v1' }
    : { apiKey: process.env.OPENAI_API_KEY }
);
setDefaultProvider(provider);

// ─────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────

function separator(title: string) {
  console.log(`\n${'═'.repeat(64)}`);
  console.log(`  ${title}`);
  console.log(`${'═'.repeat(64)}\n`);
}

function indent(text: string, prefix = '    '): string {
  return text.split('\n').map(l => prefix + l).join('\n');
}

async function pause(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 1: Race Strategy — Fastest Valid Response Wins
// ═══════════════════════════════════════════════════════════════════

async function demoRace() {
  separator('BRAID: Race Strategy — First Valid Response Wins');

  const question = 'What is the most distant object ever observed by humans?';
  console.log(`  Question: "${question}"\n`);
  console.log(`  Racing ${models.length} models...`);
  console.log(`  Models: ${models.join(', ')}\n`);

  const [input] = createSignal(question);

  // Create a synapse per model
  const synapses = models.map((model, i) => 
    createSynapse({
      model,
      signature: 'question -> answer, confidence',
      dependencies: () => ({ question: input() }),
      autoTrigger: false,
      stream: false,
      debounce: 0,
      temperature: 0.3,
    })
  );

  const startTime = Date.now();

  // Fire all in parallel
  const results = await Promise.allSettled(
    synapses.map(s => s.trigger())
  );

  const elapsed = Date.now() - startTime;

  // Create braid with race strategy
  const braid = createBraid(synapses, {
    strategy: 'race',
    validator: (output: any) => output !== undefined && output !== null,
  });

  await pause(100);

  // Display results
  console.log(`  ⏱  Total time: ${elapsed}ms\n`);
  console.log('  ┌─────────────────────────────────────────────────────────┐');

  results.forEach((result, i) => {
    const status = result.status === 'fulfilled' ? '✓' : '✗';
    const output = result.status === 'fulfilled' ? result.value : null;
    const isWinner = i === braid.winner();
    const prefix = isWinner ? '  │ 🏆' : '  │   ';
    console.log(`${prefix} [${status}] ${models[i]}`);
    if (output) {
      const answer = typeof output === 'object' && (output as any).answer 
        ? (output as any).answer 
        : JSON.stringify(output);
      console.log(`  │     ${String(answer).slice(0, 60)}${String(answer).length > 60 ? '...' : ''}`);
    }
  });

  console.log('  └─────────────────────────────────────────────────────────┘');
  console.log(`\n  Winner: ${models[braid.winner()]} (index ${braid.winner()})`);

  // Cleanup
  braid.dispose();
  synapses.forEach(s => s.dispose());
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 2: Consensus Strategy — Agreement Among Models
// ═══════════════════════════════════════════════════════════════════

async function demoConsensus() {
  separator('BRAID: Consensus Strategy — Models Must Agree');

  const question = 'Is water wet? Answer with exactly one word: yes or no.';
  console.log(`  Question: "${question}"\n`);
  console.log(`  Requiring consensus from ${models.length} models...\n`);

  const [input] = createSignal(question);

  const synapses = models.map(model =>
    createSynapse({
      model,
      signature: 'question -> answer',
      dependencies: () => ({ question: input() }),
      autoTrigger: false,
      stream: false,
      debounce: 0,
      temperature: 0.1,
    })
  );

  // Fire all
  const results = await Promise.allSettled(
    synapses.map(s => s.trigger())
  );

  // Create braid with consensus
  const braid = createBraid(synapses, {
    strategy: 'consensus',
    consensusThreshold: 0.5,
    equals: (a: any, b: any) => {
      // Normalize answers for comparison
      const normalize = (v: any) => {
        const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
        return s.toLowerCase().replace(/[^a-z]/g, '');
      };
      return normalize(a) === normalize(b);
    },
    validator: (output: any) => output !== undefined,
  });

  await pause(200);

  // Display
  console.log('  Responses:');
  results.forEach((result, i) => {
    if (result.status === 'fulfilled') {
      const val = result.value;
      const answer = typeof val === 'object' && (val as any).answer 
        ? (val as any).answer : JSON.stringify(val);
      console.log(`    ${models[i]}: "${answer}"`);
    } else {
      console.log(`    ${models[i]}: ERROR`);
    }
  });

  const consensus = braid.output();
  if (consensus) {
    console.log(`\n  ✓ Consensus reached!`);
    console.log(`    Agreed answer: ${JSON.stringify(consensus)}`);
  } else {
    console.log(`\n  ✗ No consensus reached (models disagreed)`);
  }

  braid.dispose();
  synapses.forEach(s => s.dispose());
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 3: Judge Strategy — Meta-Model Picks the Best
// ═══════════════════════════════════════════════════════════════════

async function demoJudge() {
  separator('BRAID: Judge Strategy — Pick the Best Response');

  const question = 'Explain quantum entanglement in one sentence suitable for a 10-year-old.';
  console.log(`  Question: "${question}"\n`);

  const [input] = createSignal(question);

  const synapses = models.map(model =>
    createSynapse({
      model,
      signature: 'question -> explanation',
      dependencies: () => ({ question: input() }),
      autoTrigger: false,
      stream: false,
      debounce: 0,
      temperature: 0.7,
    })
  );

  // Fire all
  await Promise.allSettled(synapses.map(s => s.trigger()));

  // Collect outputs
  const outputs: string[] = [];
  synapses.forEach((s, i) => {
    const out = s.output.peek();
    const text = typeof out === 'object' && (out as any).explanation
      ? (out as any).explanation : JSON.stringify(out);
    outputs.push(text);
    console.log(`  [${models[i]}]:`);
    console.log(indent(String(text).slice(0, 120)));
    console.log('');
  });

  // Use a simple heuristic judge (in production, this would be another LLM)
  const braid = createBraid(synapses, {
    strategy: 'judge',
    judge: (candidates) => {
      // Heuristic: prefer shorter, clearer answers (good for kids)
      let bestIdx = 0;
      let bestScore = -Infinity;
      for (const c of candidates) {
        const text = typeof c.output === 'object' && (c.output as any).explanation
          ? (c.output as any).explanation : JSON.stringify(c.output);
        // Score: penalize length, reward question marks and simple words
        const len = String(text).length;
        const score = (len < 100 ? 50 : 0) + (len < 200 ? 30 : 0) - (len / 10);
        if (score > bestScore) {
          bestScore = score;
          bestIdx = c.index;
        }
      }
      return bestIdx;
    },
    validator: (output: any) => output !== undefined,
  });

  await pause(100);

  console.log(`  ─────────────────────────────────────`);
  console.log(`  🏆 Judge selected: ${models[braid.winner()]}`);
  
  // Show Braille fingerprints for the outputs
  console.log(`\n  Braille Fingerprints (semantic similarity):`);
  const fingerprints = outputs.map(o => semanticFingerprint(String(o)));
  fingerprints.forEach((fp, i) => {
    console.log(`    ${models[i]}: ${fp}`);
  });
  
  if (fingerprints.length >= 2) {
    const sim = fingerprintSimilarity(fingerprints[0], fingerprints[1]);
    console.log(`\n    Similarity [0] vs [1]: ${(sim * 100).toFixed(1)}%`);
  }

  braid.dispose();
  synapses.forEach(s => s.dispose());
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 4: Streaming Braid — Watch Models Race in Real-Time
// ═══════════════════════════════════════════════════════════════════

async function demoStreamingRace() {
  separator('BRAID: Streaming Race — Real-Time Token Competition');

  const prompt = 'Write a haiku about artificial intelligence.';
  console.log(`  Prompt: "${prompt}"\n`);
  console.log(`  Streaming from ${models.length} models simultaneously...\n`);

  const [input] = createSignal(prompt);

  const synapses = models.map(model =>
    createSynapse({
      model,
      signature: 'prompt -> haiku',
      dependencies: () => ({ prompt: input() }),
      autoTrigger: false,
      stream: true,
      debounce: 0,
      temperature: 0.9,
    })
  );

  // Watch streams
  const streamLengths: number[] = models.map(() => 0);
  const disposers = synapses.map((synapse, i) => 
    createEffect(() => {
      const text = synapse.stream();
      if (text) {
        streamLengths[i] = text.length;
        // Show progress bars using Braille
        const maxLen = 100;
        const progress = Math.min(text.length / maxLen, 1);
        const bar = confidenceBar(progress, 10);
        process.stdout.write(`\r  ${models[i].padEnd(15)} ${bar} ${text.length} chars`);
      }
    })
  );

  // Race them
  const startTime = Date.now();
  await Promise.allSettled(synapses.map(s => s.trigger()));
  const elapsed = Date.now() - startTime;

  console.log(`\n\n  ⏱  All models finished in ${elapsed}ms\n`);

  // Show final results
  console.log('  Final outputs:');
  synapses.forEach((s, i) => {
    const out = s.output.peek();
    const text = typeof out === 'object' && (out as any).haiku
      ? (out as any).haiku : s.stream.peek();
    console.log(`\n  [${models[i]}]:`);
    console.log(indent(String(text).slice(0, 200)));
  });

  // Braille confidence display
  console.log('\n\n  Accessible State (Braille):');
  synapses.forEach((s, i) => {
    const acc = accessibleConfidence(1.0, models[i]);
    console.log(`    ${acc.braille} ${acc.ariaLabel}`);
  });

  disposers.forEach(d => d());
  synapses.forEach(s => s.dispose());
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║         Synapse.js — Multi-LLM Braid Demonstration              ║
║                                                                  ║
║  Backend: ${USE_OLLAMA ? 'Ollama (local)     ' : 'OpenAI API         '}                            ║
║  Models: ${models.join(', ').slice(0, 48).padEnd(48)}   ║
╚══════════════════════════════════════════════════════════════════╝`);

  await demoRace();
  await demoConsensus();
  await demoJudge();
  await demoStreamingRace();

  separator('BRAID DEMO COMPLETE');
  console.log('  Demonstrated:');
  console.log('    • Race: First valid response wins');
  console.log('    • Consensus: Majority agreement required');
  console.log('    • Judge: Meta-evaluation picks best');
  console.log('    • Streaming race with Braille progress bars');
  console.log('    • Semantic fingerprints for output comparison');
  console.log('');
}

main().catch(err => {
  console.error('\nError:', err.message);
  if (USE_OLLAMA) {
    console.error('Make sure Ollama is running: ollama serve');
    console.error(`And models are pulled: ${OLLAMA_MODELS.map(m => `ollama pull ${m}`).join(', ')}`);
  }
  process.exit(1);
});
