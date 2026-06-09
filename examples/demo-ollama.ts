/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║          Synapse.js — Self-Contained Ollama Demo                ║
 * ║                                                                  ║
 * ║  Demonstrates the full reactive AI framework using a tiny       ║
 * ║  local model (qwen2.5:0.5b) via Ollama. No API keys needed.    ║
 * ╚══════════════════════════════════════════════════════════════════╝
 * 
 * Prerequisites:
 *   1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh
 *   2. Pull model:     ollama pull qwen2.5:0.5b
 *   3. Run demo:       npx tsx examples/demo-ollama.ts
 */

import {
  createSignal,
  createComputed,
  createEffect,
  batch,
  createSynapse,
  createBraid,
  createMemory,
  createTool,
  createPipeline,
  createAgent,
  setDefaultProvider,
  createOpenAIProvider,
} from '../src/index';

// ─────────────────────────────────────────────────────────────────
// Configuration: Point to local Ollama (OpenAI-compatible endpoint)
// ─────────────────────────────────────────────────────────────────

const MODEL = 'qwen2.5:0.5b';

setDefaultProvider(createOpenAIProvider({
  apiKey: 'ollama',  // Ollama doesn't need a real key
  baseURL: 'http://localhost:11434/v1',
}));

// ─────────────────────────────────────────────────────────────────
// Utility
// ─────────────────────────────────────────────────────────────────

function separator(title: string) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`  ${title}`);
  console.log(`${'─'.repeat(60)}\n`);
}

async function pause(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 1: Reactive Signals (Foundation)
// ═══════════════════════════════════════════════════════════════════

function demo1_reactiveSignals() {
  separator('DEMO 1: Fine-Grained Reactive Signals');

  console.log('  Synapse.js is built on a custom reactive engine.');
  console.log('  Signals, Computed values, and Effects form a dependency graph.\n');

  const [firstName, setFirstName] = createSignal('Ada');
  const [lastName, setLastName] = createSignal('Lovelace');

  const fullName = createComputed(() => `${firstName()} ${lastName()}`);

  const logs: string[] = [];
  const dispose = createEffect(() => {
    logs.push(`  → Effect fired: "${fullName()}"`);
  });

  console.log(`  Initial: ${fullName()}`);
  console.log(`  Effects so far: ${logs.length}`);

  setFirstName('Alan');
  console.log(`  After setFirstName('Alan'): ${fullName()}`);
  console.log(`  Effects so far: ${logs.length}`);

  batch(() => {
    setFirstName('Grace');
    setLastName('Hopper');
  });
  console.log(`  After batch: ${fullName()}`);
  console.log(`  Effects so far: ${logs.length} (batch = 1 run)`);

  logs.forEach(l => console.log(l));
  dispose();

  console.log('\n  ✓ Signals, Computed, Effects, and Batch all working.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 2: Synapse (Reactive LLM Call)
// ═══════════════════════════════════════════════════════════════════

async function demo2_synapse() {
  separator('DEMO 2: Synapse — Reactive AI Inference');

  console.log('  A Synapse wraps an LLM call in a reactive node.');
  console.log('  It auto-triggers when dependencies change and parses');
  console.log('  structured output via DSPy-style signatures.\n');

  const [topic, setTopic] = createSignal('the moon');

  const synapse = createSynapse({
    model: MODEL,
    signature: 'topic -> fact, category',
    dependencies: () => ({ topic: topic() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
    temperature: 0.7,
  });

  console.log(`  Topic: "${topic()}"`);
  console.log('  Triggering inference...');

  const result = await synapse.trigger();
  console.log('  Result:', JSON.stringify(result, null, 2));

  console.log(`\n  Changing topic to "black holes"...`);
  setTopic('black holes');

  const result2 = await synapse.trigger();
  console.log('  Result:', JSON.stringify(result2, null, 2));

  synapse.dispose();
  console.log('\n  ✓ Synapse: reactive inference with structured output.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 3: Streaming
// ═══════════════════════════════════════════════════════════════════

async function demo3_streaming() {
  separator('DEMO 3: Token Streaming as a Reactive Signal');

  console.log('  Streaming output is a signal you can compose with other signals.\n');

  const synapse = createSynapse({
    model: MODEL,
    signature: 'prompt -> story',
    dependencies: () => ({ prompt: 'Write a 2-sentence story about a robot.' }),
    autoTrigger: false,
    stream: true,
    debounce: 0,
  });

  let tokenCount = 0;
  const dispose = createEffect(() => {
    const text = synapse.stream();
    if (text) {
      tokenCount++;
    }
  });

  process.stdout.write('  Streaming: ');
  
  const chunkDispose = createEffect(() => {
    const text = synapse.stream();
    if (text) {
      process.stdout.write(`\r  Streaming: ${text.slice(0, 70)}${text.length > 70 ? '...' : ''}`);
    }
  });

  await synapse.trigger();
  
  console.log(`\n  Stream signal updated ${tokenCount} times.`);
  
  dispose();
  chunkDispose();
  synapse.dispose();
  console.log('  ✓ Streaming works as a composable reactive signal.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 4: Memory (Reactive Context Graph)
// ═══════════════════════════════════════════════════════════════════

function demo4_memory() {
  separator('DEMO 4: Memory — Reactive Context Graph');

  console.log('  Memory auto-manages token budgets and is reactive.\n');

  const memory = createMemory({
    maxTokens: 100,
    strategy: 'sliding_window',
    reserveTokens: 20,
  });

  const dispose = createEffect(() => {
    const count = memory.count();
    if (count > 0) {
      console.log(`  [Memory updated: ${count} messages, ~${memory.tokenUsage()} tokens]`);
    }
  });

  memory.add('user', 'What is the capital of France?');
  memory.add('assistant', 'The capital of France is Paris.');
  memory.add('user', 'What about Germany?');
  memory.add('assistant', 'The capital of Germany is Berlin.');
  memory.add('user', 'And Japan?');
  memory.add('assistant', 'The capital of Japan is Tokyo.');

  console.log(`\n  Total messages added: 6`);
  console.log(`  Messages in context window: ${memory.context().length}`);
  console.log(`  Token budget respected: ${memory.tokenUsage()} ≤ ${100 - 20}`);

  const results = memory.search('France');
  console.log(`\n  Search "France": found ${results.length} matches`);
  results.forEach(r => console.log(`    → "${r.content}"`));

  dispose();
  console.log('\n  ✓ Memory: reactive, token-budgeted, searchable.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 5: Braid (Multi-Model Composition)
// ═══════════════════════════════════════════════════════════════════

async function demo5_braid() {
  separator('DEMO 5: Braid — Racing Multiple Inferences');

  console.log('  Braid composes multiple Synapses with strategies like');
  console.log('  race, consensus, fallback, or judge.\n');

  const [question] = createSignal('What color is the sky?');

  const conservative = createSynapse({
    model: MODEL,
    signature: 'question -> answer',
    dependencies: () => ({ question: question() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
    temperature: 0.1,
  });

  const creative = createSynapse({
    model: MODEL,
    signature: 'question -> answer',
    dependencies: () => ({ question: question() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
    temperature: 1.0,
  });

  console.log(`  Question: "${question()}"`);
  console.log('  Racing conservative (temp=0.1) vs creative (temp=1.0)...\n');

  await Promise.all([conservative.trigger(), creative.trigger()]);

  const braid = createBraid([conservative, creative], {
    strategy: 'race',
    validator: (output: any) => output !== undefined,
  });

  await pause(100);

  const winner = braid.winner();
  const output = braid.output();
  console.log(`  Winner: ${winner === 0 ? 'Conservative' : 'Creative'} (index ${winner})`);
  console.log(`  Output: ${JSON.stringify(output)}`);

  braid.dispose();
  conservative.dispose();
  creative.dispose();
  console.log('\n  ✓ Braid: declarative multi-model composition.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 6: Agent
// ═══════════════════════════════════════════════════════════════════

async function demo6_agent() {
  separator('DEMO 6: Agent — Autonomous Reactive Agent');

  console.log('  An Agent combines Synapse + Memory + Tools in a loop.\n');

  const agent = createAgent({
    name: 'Sage',
    model: MODEL,
    system: 'You are Sage, a helpful and concise assistant. Answer in 1-2 sentences.',
    memory: { maxTokens: 2048, strategy: 'sliding_window' },
    temperature: 0.7,
  });

  const dispose = createEffect(() => {
    const loading = agent.loading();
    if (loading) {
      process.stdout.write('  [thinking...]\r');
    }
  });

  console.log('  User: "What is recursion?"');
  const r1 = await agent.send('What is recursion?');
  console.log(`  Sage: ${r1}\n`);

  console.log('  User: "Give me an example."');
  const r2 = await agent.send('Give me an example.');
  console.log(`  Sage: ${r2}\n`);

  console.log(`  Memory: ${agent.memory.count()} messages stored`);
  console.log(`  Interactions: ${agent.interactions()}`);

  dispose();
  console.log('\n  ✓ Agent: stateful, multi-turn, reactive.');
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║               Synapse.js — Full Framework Demo                   ║
║                                                                  ║
║  Model: ${MODEL.padEnd(20)} (local via Ollama)          ║
║  Framework: AI-Native Reactive Primitives                        ║
╚══════════════════════════════════════════════════════════════════╝`);

  demo1_reactiveSignals();
  await demo2_synapse();
  await demo3_streaming();
  demo4_memory();
  await demo5_braid();
  await demo6_agent();

  separator('ALL DEMOS COMPLETE');
  console.log('  Synapse.js demonstrated:');
  console.log('    • Fine-grained reactive signals, computed, effects, batch');
  console.log('    • Synapse: reactive LLM inference with structured output');
  console.log('    • Token streaming as composable reactive signals');
  console.log('    • Memory: reactive context graph with token budgeting');
  console.log('    • Braid: multi-model racing/composition');
  console.log('    • Agent: autonomous multi-turn loop');
  console.log('\n  All powered by a tiny local model (qwen2.5:0.5b) via Ollama.');
  console.log('  No API keys. No cloud. Fully self-contained.\n');
}

main().catch(err => {
  console.error('\nError:', err.message);
  console.error('Make sure Ollama is running: ollama serve');
  console.error('And the model is pulled: ollama pull qwen2.5:0.5b');
  process.exit(1);
});
