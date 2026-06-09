/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║     Synapse.js — Streaming vs Normal: Dual-Mode Braid          ║
 * ║                                                                  ║
 * ║  Races streaming and non-streaming calls side by side,          ║
 * ║  demonstrating that they CAN produce different results          ║
 * ║  and showing the streaming_race strategy in action.             ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * Prerequisites:
 *   ollama pull qwen2.5:0.5b
 *   npx tsx examples/demo-streaming-vs-normal.ts
 */

import {
  createSignal,
  createEffect,
  createSynapse,
  createBraid,
  setDefaultProvider,
  createOpenAIProvider,
  confidenceBar,
  semanticFingerprint,
  fingerprintSimilarity,
} from '../src/index';

// ─────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────

const MODEL = process.env.MODEL || 'qwen2.5:0.5b';

const provider = createOpenAIProvider(
  process.env.OPENAI_API_KEY
    ? { apiKey: process.env.OPENAI_API_KEY }
    : { apiKey: 'ollama', baseURL: 'http://localhost:11434/v1' }
);
setDefaultProvider(provider);

function separator(title: string) {
  console.log(`\n${'═'.repeat(64)}`);
  console.log(`  ${title}`);
  console.log(`${'═'.repeat(64)}\n`);
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 1: Same prompt, streaming vs non-streaming — do they differ?
// ═══════════════════════════════════════════════════════════════════

async function demoDifference() {
  separator('EXPERIMENT: Same Model, Same Prompt — Streaming vs Normal');

  const prompt = 'In exactly 10 words, describe what makes the ocean blue.';
  console.log(`  Prompt: "${prompt}"`);
  console.log(`  Model:  ${MODEL}`);
  console.log(`  Hypothesis: Streaming and non-streaming MAY give different results.\n`);

  const [input] = createSignal(prompt);

  const streamingSynapse = createSynapse({
    model: MODEL,
    signature: 'prompt -> response',
    dependencies: () => ({ prompt: input() }),
    autoTrigger: false,
    stream: true,
    debounce: 0,
    temperature: 0.7,
    maxTokens: 60,
  });

  const normalSynapse = createSynapse({
    model: MODEL,
    signature: 'prompt -> response',
    dependencies: () => ({ prompt: input() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
    temperature: 0.7,
    maxTokens: 60,
  });

  // Track streaming progress
  let streamChunks = 0;
  const disposeStreamWatch = createEffect(() => {
    const text = streamingSynapse.stream();
    if (text) streamChunks++;
  });

  console.log('  Running both simultaneously...\n');
  const start = Date.now();

  const [streamResult, normalResult] = await Promise.all([
    streamingSynapse.trigger(),
    normalSynapse.trigger(),
  ]);

  const elapsed = Date.now() - start;

  const streamText = typeof streamResult === 'object' && (streamResult as any).response
    ? (streamResult as any).response : JSON.stringify(streamResult);
  const normalText = typeof normalResult === 'object' && (normalResult as any).response
    ? (normalResult as any).response : JSON.stringify(normalResult);

  console.log(`  ┌─── Streaming (${streamChunks} chunks) ──────────────────────────────┐`);
  console.log(`  │  "${String(streamText).slice(0, 60)}"`);
  console.log(`  └─────────────────────────────────────────────────────────┘`);
  console.log(`  ┌─── Non-Streaming ────────────────────────────────────────┐`);
  console.log(`  │  "${String(normalText).slice(0, 60)}"`);
  console.log(`  └─────────────────────────────────────────────────────────┘`);

  // Compare
  const fp1 = semanticFingerprint(String(streamText), 8);
  const fp2 = semanticFingerprint(String(normalText), 8);
  const similarity = fingerprintSimilarity(fp1, fp2);
  const identical = String(streamText).trim() === String(normalText).trim();

  console.log(`\n  Results:`);
  console.log(`    Identical text?  ${identical ? '✓ YES' : '✗ NO (different!)'}`);
  console.log(`    Fingerprint sim: ${(similarity * 100).toFixed(1)}%`);
  console.log(`    Stream FP:       ${fp1}`);
  console.log(`    Normal FP:       ${fp2}`);
  console.log(`    Time:            ${elapsed}ms`);

  if (!identical) {
    console.log(`\n  ⚡ This proves that streaming and non-streaming are INDEPENDENT`);
    console.log(`     inference calls. Even with the same model/prompt/temperature,`);
    console.log(`     token sampling is stochastic. Each call rolls the dice anew.`);
  }

  disposeStreamWatch();
  streamingSynapse.dispose();
  normalSynapse.dispose();
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 2: streaming_race strategy — first to finish streaming wins
// ═══════════════════════════════════════════════════════════════════

async function demoStreamingRace() {
  separator('STREAMING RACE: First Model to Finish Streaming Wins');

  const prompt = 'Name 3 prime numbers greater than 100.';
  console.log(`  Prompt: "${prompt}"`);
  console.log(`  Strategy: streaming_race — winner is whoever finishes streaming first\n`);

  const [input] = createSignal(prompt);

  // Create multiple synapses — all streaming, same model but different temps
  // (simulates racing different "approaches")
  const configs = [
    { label: 'Conservative (t=0.1)', temperature: 0.1 },
    { label: 'Balanced (t=0.5)', temperature: 0.5 },
    { label: 'Creative (t=1.0)', temperature: 1.0 },
  ];

  const synapses = configs.map(c =>
    createSynapse({
      model: MODEL,
      signature: 'prompt -> answer',
      dependencies: () => ({ prompt: input() }),
      autoTrigger: false,
      stream: true,
      debounce: 0,
      temperature: c.temperature,
      maxTokens: 50,
    })
  );

  // Create a streaming_race braid
  const braid = createBraid(synapses, {
    strategy: 'streaming_race',
    validator: (output: any) => output !== undefined,
    timeout: 15000,
  });

  // Live progress display
  const streamStates: string[] = configs.map(() => '');
  const disposers = synapses.map((s, i) =>
    createEffect(() => {
      const text = s.stream();
      if (text) {
        streamStates[i] = text;
        const bar = confidenceBar(Math.min(text.length / 80, 1), 8);
        process.stdout.write(`\r  ${configs[i].label.padEnd(22)} ${bar} ${text.length} chars`);
      }
    })
  );

  console.log('  Racing...\n');
  const start = Date.now();

  // Use braid.trigger() — fires all and resolves via streaming_race
  const result = await braid.trigger();
  const elapsed = Date.now() - start;

  console.log(`\n\n  ⏱  Race completed in ${elapsed}ms`);
  console.log(`  🏆 Winner: ${configs[braid.winner()].label} (index ${braid.winner()})\n`);

  // Show all final outputs
  const timingValues = braid.timings();
  configs.forEach((c, i) => {
    const text = typeof synapses[i].output.peek() === 'object'
      ? (synapses[i].output.peek() as any).answer
      : streamStates[i];
    const isWinner = i === braid.winner();
    const timing = timingValues[i] >= 0 ? `${timingValues[i]}ms` : 'n/a';
    const prefix = isWinner ? '  🏆' : '    ';
    console.log(`${prefix} [${timing}] ${c.label}:`);
    console.log(`       "${String(text).slice(0, 70)}"`);
  });

  disposers.forEach(d => d());
  braid.dispose();
  synapses.forEach(s => s.dispose());
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 3: Dual-mode braid — streaming AND normal in one race
// ═══════════════════════════════════════════════════════════════════

async function demoDualMode() {
  separator('DUAL-MODE BRAID: Streaming + Normal in One Race');

  const prompt = 'What is the fastest land animal? One sentence.';
  console.log(`  Prompt: "${prompt}"`);
  console.log(`  Racing: 1 streaming + 1 non-streaming (same model)`);
  console.log(`  Question: Does streaming or non-streaming resolve first?\n`);

  const [input] = createSignal(prompt);

  const streamingSynapse = createSynapse({
    model: MODEL,
    signature: 'prompt -> answer',
    dependencies: () => ({ prompt: input() }),
    autoTrigger: false,
    stream: true,
    debounce: 0,
    temperature: 0.3,
    maxTokens: 50,
  });

  const normalSynapse = createSynapse({
    model: MODEL,
    signature: 'prompt -> answer',
    dependencies: () => ({ prompt: input() }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
    temperature: 0.3,
    maxTokens: 50,
  });

  // Live streaming indicator
  const disposeWatch = createEffect(() => {
    const text = streamingSynapse.stream();
    if (text) {
      const bar = confidenceBar(Math.min(text.length / 60, 1), 12);
      process.stdout.write(`\r  Streaming: ${bar} ${text.length} chars`);
    }
  });

  const braid = createBraid([streamingSynapse, normalSynapse], {
    strategy: 'streaming_race',
    validator: (output: any) => output !== undefined,
  });

  const start = Date.now();
  const result = await braid.trigger();
  const elapsed = Date.now() - start;

  const winnerLabel = braid.winner() === 0 ? 'STREAMING' : 'NON-STREAMING';
  const timingValues = braid.timings();

  console.log(`\n\n  Result in ${elapsed}ms:`);
  console.log(`  Winner: ${winnerLabel}\n`);

  console.log(`  ┌─── Streaming ─────────────────────────────────────────────┐`);
  const sOut = streamingSynapse.output.peek();
  const sText = typeof sOut === 'object' && (sOut as any).answer ? (sOut as any).answer : streamingSynapse.stream.peek();
  console.log(`  │  "${String(sText).slice(0, 55)}"`);
  console.log(`  │  Time: ${timingValues[0] >= 0 ? timingValues[0] + 'ms' : 'n/a'}`);
  console.log(`  └─────────────────────────────────────────────────────────┘`);

  console.log(`  ┌─── Non-Streaming ─────────────────────────────────────────┐`);
  const nOut = normalSynapse.output.peek();
  const nText = typeof nOut === 'object' && (nOut as any).answer ? (nOut as any).answer : JSON.stringify(nOut);
  console.log(`  │  "${String(nText).slice(0, 55)}"`);
  console.log(`  │  Time: ${timingValues[1] >= 0 ? timingValues[1] + 'ms' : 'n/a'}`);
  console.log(`  └─────────────────────────────────────────────────────────┘`);

  // Were they different?
  const identical = String(sText).trim() === String(nText).trim();
  console.log(`\n  Same answer? ${identical ? '✓ YES' : '✗ NO — independent inference calls!'}`);

  disposeWatch();
  braid.dispose();
  streamingSynapse.dispose();
  normalSynapse.dispose();
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 4: Repeated trials — how often do streaming/normal differ?
// ═══════════════════════════════════════════════════════════════════

async function demoTrials() {
  separator('TRIAL: How Often Do Streaming vs Normal Differ?');

  const prompt = 'What is 7 * 8? Just the number.';
  const trials = 5;
  console.log(`  Prompt: "${prompt}"`);
  console.log(`  Running ${trials} trials, comparing streaming vs normal output\n`);

  let matches = 0;
  let diffs = 0;

  for (let trial = 1; trial <= trials; trial++) {
    const [input] = createSignal(prompt);

    const streaming = createSynapse({
      model: MODEL,
      signature: 'prompt -> answer',
      dependencies: () => ({ prompt: input() }),
      autoTrigger: false,
      stream: true,
      debounce: 0,
      temperature: 0.1,
      maxTokens: 20,
    });

    const normal = createSynapse({
      model: MODEL,
      signature: 'prompt -> answer',
      dependencies: () => ({ prompt: input() }),
      autoTrigger: false,
      stream: false,
      debounce: 0,
      temperature: 0.1,
      maxTokens: 20,
    });

    const [sResult, nResult] = await Promise.all([
      streaming.trigger(),
      normal.trigger(),
    ]);

    const sText = typeof sResult === 'object' && (sResult as any).answer
      ? (sResult as any).answer : JSON.stringify(sResult);
    const nText = typeof nResult === 'object' && (nResult as any).answer
      ? (nResult as any).answer : JSON.stringify(nResult);

    const same = String(sText).trim() === String(nText).trim();
    if (same) matches++;
    else diffs++;

    const icon = same ? '=' : '≠';
    console.log(`    Trial ${trial}: stream="${String(sText).slice(0,20).trim()}" ${icon} normal="${String(nText).slice(0,20).trim()}"`);

    streaming.dispose();
    normal.dispose();
  }

  console.log(`\n  Results over ${trials} trials:`);
  console.log(`    Identical: ${matches}/${trials} (${(matches/trials*100).toFixed(0)}%)`);
  console.log(`    Different: ${diffs}/${trials} (${(diffs/trials*100).toFixed(0)}%)`);
  
  if (diffs > 0) {
    console.log(`\n  ⚡ Even at temperature=0.1, streaming and non-streaming diverge.`);
    console.log(`     This is because they're independent API calls with separate`);
    console.log(`     random seeds. The only guarantee of identical output is temp=0`);
    console.log(`     with a deterministic backend (which Ollama is NOT guaranteed to be).`);
  } else {
    console.log(`\n  ✓ With very low temperature and simple prompts, results often match.`);
    console.log(`    But this is not guaranteed — try with temperature=0.7 to see divergence.`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║    Synapse.js — Streaming vs Normal: Dual-Mode Braid Demo       ║
║                                                                  ║
║  Can streaming and non-streaming give different results?         ║
║  YES — and here's proof, plus a streaming_race strategy.         ║
║                                                                  ║
║  Model: ${MODEL.padEnd(20)}                                     ║
╚══════════════════════════════════════════════════════════════════╝`);

  // Warmup
  console.log('\n  Warming up model...');
  const warmup = createSynapse({
    model: MODEL,
    signature: 'x -> y',
    dependencies: () => ({ x: 'hi' }),
    autoTrigger: false,
    stream: false,
    debounce: 0,
    maxTokens: 5,
    temperature: 0,
  });
  await warmup.trigger().catch(() => {});
  warmup.dispose();
  console.log('  ✓ Ready\n');

  await demoDifference();
  await demoStreamingRace();
  await demoDualMode();
  await demoTrials();

  separator('DEMO COMPLETE');
  console.log('  Key takeaways:');
  console.log('');
  console.log('  1. Streaming and non-streaming ARE independent inference calls');
  console.log('     → Different random seeds → potentially different outputs');
  console.log('');
  console.log('  2. streaming_race strategy resolves when the first model');
  console.log('     FINISHES streaming (not just starts) — true race semantics');
  console.log('');
  console.log('  3. You can mix streaming and non-streaming in a single braid');
  console.log('     to get the best of both worlds: fast resolution + live feedback');
  console.log('');
  console.log('  4. For deterministic output, use temperature=0 — but even then,');
  console.log('     some backends (Ollama, vLLM) may not guarantee identical results');
  console.log('');
}

main().catch(err => {
  console.error('\nError:', err.message);
  console.error('Make sure Ollama is running and model is pulled: ollama pull ' + MODEL);
  process.exit(1);
});
