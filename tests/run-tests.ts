/**
 * Synapse.js Test Suite
 * Run with: npx tsx tests/run-tests.ts
 */

import {
  createSignal,
  createComputed,
  createEffect,
  createAsyncSignal,
  batch,
  createSynapse,
  createBraid,
  createMemory,
  createTool,
  createPipeline,
  setDefaultProvider,
  createMockProvider,
  parseSignature,
  // Braille
  byteToBraille,
  brailleToByte,
  encodeToBraille,
  decodeBrailleToString,
  confidenceToBraille,
  confidenceBar,
  semanticFingerprint,
  fingerprintSimilarity,
} from '../src/index';

let passed = 0;
let failed = 0;
let currentSuite = '';

function describe(name: string, fn: () => void | Promise<void>) {
  currentSuite = name;
  console.log(`\n━━━ ${name} ━━━`);
  return fn();
}

function assert(condition: boolean, msg: string) {
  if (condition) {
    passed++;
    console.log(`  ✓ ${msg}`);
  } else {
    failed++;
    console.log(`  ✗ FAIL: ${msg}`);
  }
}

async function asyncDescribe(name: string, fn: () => Promise<void>) {
  currentSuite = name;
  console.log(`\n━━━ ${name} ━━━`);
  await fn();
}

// ============================================================
// REACTIVE ENGINE TESTS
// ============================================================

describe('Signal', () => {
  const [count, setCount] = createSignal(0);
  assert(count() === 0, 'holds initial value');
  setCount(5);
  assert(count() === 5, 'updates on set');
  setCount((prev: number) => prev + 1);
  assert(count() === 6, 'functional update');

  const [obj, setObj] = createSignal(
    { x: 1 },
    { equals: (a, b) => a.x === b.x }
  );
  let objEffectCount = 0;
  createEffect(() => { obj(); objEffectCount++; });
  setObj({ x: 1 });
  assert(objEffectCount === 1, 'custom equality prevents spurious updates');
  setObj({ x: 2 });
  assert(objEffectCount === 2, 'custom equality allows real updates');
});

describe('Computed', () => {
  const [a, setA] = createSignal(2);
  const [b, setB] = createSignal(3);
  const sum = createComputed(() => a() + b());
  assert(sum() === 5, 'derives from signals');
  setA(10);
  assert(sum() === 13, 'updates when deps change');

  const doubled = createComputed(() => sum() * 2);
  assert(doubled() === 26, 'chains computeds');
  setB(7);
  assert(doubled() === 34, 'chain updates propagate');

  let computeCount = 0;
  const [x, setX] = createSignal(1);
  const lazy = createComputed(() => { computeCount++; return x() * 2; });
  assert(computeCount === 0, 'lazy: not computed until read');
  lazy();
  assert(computeCount === 1, 'lazy: computed on first read');
  lazy();
  assert(computeCount === 1, 'lazy: cached on subsequent reads');
});

describe('Effect', () => {
  const [val, setVal] = createSignal(0);
  const values: number[] = [];
  const dispose = createEffect(() => { values.push(val()); });
  assert(values.length === 1 && values[0] === 0, 'runs immediately');
  setVal(1);
  assert(values[1] === 1, 're-runs on change');
  dispose();
  setVal(2);
  assert(values.length === 2, 'disposed effect stops');

  let cleanups = 0;
  const [c, setC] = createSignal(0);
  createEffect(() => { c(); return () => { cleanups++; }; });
  setC(1);
  assert(cleanups === 1, 'cleanup runs on re-execution');

  const [flag, setFlag] = createSignal(true);
  const [aa, setAA] = createSignal('A');
  const [bb, setBB] = createSignal('B');
  const results: string[] = [];
  createEffect(() => { results.push(flag() ? aa() : bb()); });
  assert(results[0] === 'A', 'dynamic deps: initial');
  setBB('B2');
  assert(results.length === 1, 'dynamic deps: untracked dep no-op');
  setFlag(false);
  assert(results[results.length - 1] === 'B2', 'dynamic deps: switches');
});

describe('Batch', () => {
  const [x, setX] = createSignal(1);
  const [y, setY] = createSignal(2);
  let runs = 0;
  createEffect(() => { x(); y(); runs++; });
  assert(runs === 1, 'initial run');
  batch(() => { setX(10); setY(20); });
  assert(runs === 2, 'batched: single effect run');
});

describe('AsyncSignal', () => {
  const as = createAsyncSignal<string>('init');
  assert(as.state().value === 'init', 'initial value');
  as.startLoading();
  assert(as.state().loading === true, 'loading state');
  as.resolve('done');
  assert(as.state().value === 'done', 'resolved value');
  assert(as.state().loading === false, 'loading cleared');

  const as2 = createAsyncSignal<string>();
  as2.startStreaming();
  as2.stream('hello', (prev, chunk) => (prev || '') + chunk);
  as2.stream(' world', (prev, chunk) => (prev || '') + chunk);
  assert(as2.state().value === 'hello world', 'streaming accumulates');

  const as3 = createAsyncSignal<string>();
  as3.reject(new Error('oops'));
  assert(as3.state().error?.message === 'oops', 'error state');
});

// ============================================================
// AI PRIMITIVE TESTS
// ============================================================

describe('parseSignature', () => {
  const sig = parseSignature('context, query -> answer, confidence');
  assert(sig.inputs.length === 2, 'parses inputs');
  assert(sig.outputs.length === 2, 'parses outputs');
  assert(sig.inputs[0] === 'context', 'correct input names');
  assert(sig.outputs[1] === 'confidence', 'correct output names');

  let threw = false;
  try { parseSignature('no arrow'); } catch { threw = true; }
  assert(threw, 'throws on invalid signature');
});

// Set up mock provider for AI tests
setDefaultProvider(createMockProvider({
  generator: (req) => {
    const userMsg = req.messages.find(m => m.role === 'user')?.content || '';
    if (userMsg.includes('positive')) {
      return '{"sentiment": "positive", "confidence": 0.95}';
    }
    return '{"sentiment": "neutral", "confidence": 0.5}';
  },
  delay: 10,
  chunkSize: 10,
}));

async function main() {
await asyncDescribe('Synapse', async () => {
  const [input] = createSignal('This is positive text');
  const synapse = createSynapse({
    model: 'mock',
    signature: 'text -> sentiment, confidence',
    dependencies: () => ({ text: input() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
  });

  const result = await synapse.trigger();
  assert(result !== undefined, 'returns a result');
  assert((result as any).sentiment === 'positive', 'parses structured output');
  assert((result as any).confidence === 0.95, 'parses numeric fields');
  synapse.dispose();

  const synapse2 = createSynapse({
    model: 'mock',
    signature: 'text -> sentiment',
    dependencies: () => ({ text: 'positive' }),
    autoTrigger: false,
    stream: true,
    debounce: 0,
  });

  const streamValues: string[] = [];
  const disposeWatch = createEffect(() => {
    const s = synapse2.stream();
    if (s) streamValues.push(s);
  });

  await synapse2.trigger();
  assert(streamValues.length > 0, 'streaming produces updates');
  disposeWatch();
  synapse2.dispose();
});

await asyncDescribe('Braid', async () => {
  const [input] = createSignal('test');

  const s1 = createSynapse({
    model: 'fast',
    signature: 'text -> answer',
    dependencies: () => ({ text: input() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
  });

  const s2 = createSynapse({
    model: 'slow',
    signature: 'text -> answer',
    dependencies: () => ({ text: input() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
  });

  await Promise.all([s1.trigger(), s2.trigger()]);

  const braid = createBraid([s1, s2], {
    strategy: 'race',
    validator: (output: any) => output !== undefined,
  });

  await new Promise(r => setTimeout(r, 50));
  assert(braid.output() !== undefined, 'race strategy resolves');
  assert(braid.winner() >= 0, 'identifies winner');

  braid.dispose();
  s1.dispose();
  s2.dispose();
});

describe('Memory', () => {
  const memory = createMemory({
    maxTokens: 4096,
    strategy: 'sliding_window',
  });

  memory.add('user', 'Hello!');
  memory.add('assistant', 'Hi there!');
  assert(memory.count() === 2, 'stores messages');
  assert(memory.messages()[0].content === 'Hello!', 'preserves content');

  const msgs = memory.toMessages();
  assert(msgs[0].role === 'user', 'formats for LLM');
  assert(msgs[1].content === 'Hi there!', 'formats content');

  const smallMemory = createMemory({
    maxTokens: 50,
    strategy: 'sliding_window',
    reserveTokens: 10,
  });
  for (let i = 0; i < 20; i++) {
    smallMemory.add('user', `Message ${i} with extra padding text here`);
  }
  const ctx = smallMemory.context();
  const totalTokens = ctx.reduce((sum, m) => sum + m.tokens, 0);
  assert(totalTokens <= 40, 'respects token budget');

  const mem2 = createMemory({ maxTokens: 4096, strategy: 'sliding_window' });
  const counts: number[] = [];
  createEffect(() => { counts.push(mem2.count()); });
  mem2.add('user', 'test');
  assert(counts.length === 2, 'memory is reactive');

  memory.add('user', 'Tell me about JavaScript');
  const results = memory.search('JavaScript');
  assert(results.length > 0, 'search finds matches');
});

await asyncDescribe('Tool', async () => {
  const calc = createTool({
    name: 'calculate',
    description: 'Math',
    parameters: { type: 'object', properties: { expr: { type: 'string' } } },
    execute: async ({ expr }: { expr: string }) => eval(expr),
  });

  const result = await calc.invoke({ expr: '2 + 2' });
  assert(result === 4, 'tool executes');
  assert(calc.callCount() === 1, 'tracks call count');
  assert(calc.lastResult() === 4, 'stores last result');
});

await asyncDescribe('Pipeline', async () => {
  setDefaultProvider(createMockProvider({
    generator: () => '{"result": "processed", "score": 0.9}',
    delay: 5,
  }));

  const [doc] = createSignal('test document');
  const pipeline = createPipeline({
    input: () => ({ document: doc() }),
    steps: [
      { name: 'step1', model: 'mock', signature: 'document -> result, score' },
      { name: 'step2', model: 'mock', signature: 'result, score -> result, score' },
    ],
    autoTrigger: false,
  });

  const result = await pipeline.trigger();
  assert(result !== undefined, 'pipeline produces output');
  assert(pipeline.intermediates().size === 2, 'tracks intermediates');
  assert(pipeline.intermediates().has('step1'), 'has step1 result');
  assert(pipeline.intermediates().has('step2'), 'has step2 result');
  pipeline.dispose();
});

// ============================================================
// BRAILLE ENCODING TESTS
// ============================================================

describe('Braille Encoding', () => {
  // Basic encode/decode
  assert(byteToBraille(0) === '⠀', 'byte 0 → empty braille');
  assert(byteToBraille(255) === '⣿', 'byte 255 → full braille');
  assert(brailleToByte('⠀') === 0, 'decode empty braille → 0');
  assert(brailleToByte('⣿') === 255, 'decode full braille → 255');

  // Roundtrip
  const text = 'Hello!';
  const encoded = encodeToBraille(text);
  const decoded = decodeBrailleToString(encoded);
  assert(decoded === text, 'text roundtrip through braille');

  // Confidence bar
  const bar0 = confidenceBar(0, 4);
  const bar1 = confidenceBar(1, 4);
  assert(bar0.length === 4, 'confidence bar has correct width');
  assert(bar1 === '⣿⣿⣿⣿', 'full confidence = all full braille');

  // Fingerprints
  const fp1 = semanticFingerprint('hello world', 8);
  const fp2 = semanticFingerprint('hello world', 8);
  const fp3 = semanticFingerprint('completely different text', 8);
  assert(fp1 === fp2, 'same text → same fingerprint');
  assert(fp1 !== fp3, 'different text → different fingerprint');
  assert(fingerprintSimilarity(fp1, fp2) === 1.0, 'identical fingerprints = 1.0');
  assert(fingerprintSimilarity(fp1, fp3) < 1.0, 'different fingerprints < 1.0');
});

// ============================================================
// RESULTS
// ============================================================

console.log(`\n${'═'.repeat(50)}`);
console.log(`  RESULTS: ${passed} passed, ${failed} failed`);
console.log(`${'═'.repeat(50)}\n`);

process.exit(failed > 0 ? 1 : 0);
}

main();
