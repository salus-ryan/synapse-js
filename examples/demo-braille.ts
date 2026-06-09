/**
 * ╔══════════════════════════════════════════════════════════════════╗
 * ║     Synapse.js — 8-Dot Braille Accessibility Demo               ║
 * ║                                                                  ║
 * ║  Demonstrates how Synapse.js uses 8-dot Braille (U+2800-28FF)  ║
 * ║  to make AI state simultaneously visual AND tactile.            ║
 * ║                                                                  ║
 * ║  This is what "AI-native + accessible-native" looks like.       ║
 * ╚══════════════════════════════════════════════════════════════════╝
 *
 * Run: npx tsx examples/demo-braille.ts
 */

import {
  createSignal,
  createEffect,
  createComputed,
  createSynapse,
  createBraid,
  setDefaultProvider,
  createOpenAIProvider,
  createMockProvider,
  // Braille encoding
  byteToBraille,
  brailleToByte,
  encodeToBraille,
  decodeBrailleToString,
  confidenceToBraille,
  confidenceBar,
  streamingIndicator,
  nodeStateToBraille,
  graphToBraille,
  accessibleConfidence,
  accessibleLoadingState,
  createBrailleSignal,
  semanticFingerprint,
  fingerprintSimilarity,
  BRAILLE_FULL,
  BRAILLE_EMPTY,
  type BrailleNodeState,
} from '../src/index';

// ─────────────────────────────────────────────────────────────────

function separator(title: string) {
  console.log(`\n${'═'.repeat(64)}`);
  console.log(`  ${title}`);
  console.log(`${'═'.repeat(64)}\n`);
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 1: Basic Braille Encoding
// ═══════════════════════════════════════════════════════════════════

function demo1_encoding() {
  separator('8-Dot Braille: Basic Encoding');

  console.log('  8-dot Braille uses Unicode U+2800–U+28FF (256 chars).');
  console.log('  Each character encodes one byte — 8 dots = 8 bits.\n');

  // Show the encoding
  console.log('  Text → Braille → Text roundtrip:');
  const text = 'Hello, AI!';
  const encoded = encodeToBraille(text);
  const decoded = decodeBrailleToString(encoded);
  console.log(`    Original:  "${text}"`);
  console.log(`    Braille:   "${encoded}"`);
  console.log(`    Decoded:   "${decoded}"`);
  console.log(`    Match:     ${text === decoded ? '✓' : '✗'}\n`);

  // Show byte → braille mapping
  console.log('  Byte → Braille mapping (sample):');
  const samples = [0, 1, 42, 127, 128, 200, 255];
  for (const byte of samples) {
    const char = byteToBraille(byte);
    const bits = byte.toString(2).padStart(8, '0');
    console.log(`    ${byte.toString().padStart(3)} (${bits}) → ${char}  (U+${(0x2800 + byte).toString(16).toUpperCase()})`);
  }

  console.log('\n  Key insight: A Braille display renders these as TACTILE patterns.');
  console.log('  Sighted users see visual density. Blind users feel data structure.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 2: Confidence Visualization
// ═══════════════════════════════════════════════════════════════════

function demo2_confidence() {
  separator('8-Dot Braille: AI Confidence as Tactile Patterns');

  console.log('  Confidence values rendered as Braille bars:');
  console.log('  (Screen readers announce these; Braille displays show them)\n');

  const levels = [0, 0.1, 0.25, 0.5, 0.7, 0.85, 0.95, 1.0];
  
  for (const level of levels) {
    const bar = confidenceBar(level, 10);
    const acc = accessibleConfidence(level, 'Model confidence');
    const pct = (level * 100).toFixed(0).padStart(3);
    console.log(`    ${pct}%  ${bar}  aria: "${acc.ariaLabel}"`);
  }

  console.log('\n  Single-cell confidence (one char per level):');
  for (let i = 0; i <= 8; i++) {
    const conf = i / 8;
    const char = confidenceToBraille(conf);
    console.log(`    ${(conf * 100).toFixed(0).padStart(3)}% → ${char}`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 3: Reactive Graph State
// ═══════════════════════════════════════════════════════════════════

function demo3_graphState() {
  separator('8-Dot Braille: Reactive Graph as Tactile Dashboard');

  console.log('  Each Braille char encodes a reactive node\'s full state:');
  console.log('  Bit 1=resolved, Bit 2=active, Bit 3=error, Bits 5-8=progress\n');

  const states: BrailleNodeState[] = [
    { id: 'input', active: false, progress: 1.0, error: false, resolved: true },
    { id: 'synapse1', active: true, progress: 0.5, error: false, resolved: false },
    { id: 'synapse2', active: true, progress: 0.3, error: false, resolved: false },
    { id: 'braid', active: false, progress: 0, error: false, resolved: false },
    { id: 'output', active: false, progress: 0, error: false, resolved: false },
  ];

  console.log('  Graph topology (5 nodes):');
  console.log(`    input → synapse1 ─┐`);
  console.log(`                      ├→ braid → output`);
  console.log(`    input → synapse2 ─┘\n`);

  console.log('  Braille encoding of graph state:');
  const brailleGraph = graphToBraille(states);
  console.log(`    "${brailleGraph}"\n`);

  console.log('  Individual node states:');
  states.forEach(s => {
    const char = nodeStateToBraille(s);
    console.log(`    ${char}  ${s.id.padEnd(10)} active=${s.active} progress=${s.progress} resolved=${s.resolved} error=${s.error}`);
  });

  console.log('\n  A blind user touching a Braille display feels:');
  console.log('  "One resolved node, two active at different progress, two waiting"');
  console.log('  This is REAL-TIME state communicated through TOUCH.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 4: Streaming Indicator
// ═══════════════════════════════════════════════════════════════════

async function demo4_streaming() {
  separator('8-Dot Braille: Streaming Animation');

  console.log('  Animated streaming indicator (Braille "wave"):');
  console.log('  (On a Braille display, this creates a moving tactile pattern)\n');

  for (let frame = 0; frame < 16; frame++) {
    const indicator = streamingIndicator(frame, 8);
    process.stdout.write(`\r    Frame ${frame.toString().padStart(2)}: ${indicator}`);
    await new Promise(r => setTimeout(r, 150));
  }
  console.log('\n');

  console.log('  Loading states as accessible Braille:');
  const loadingStates = [
    { loading: true, streaming: false },
    { loading: false, streaming: true, progress: 0.3 },
    { loading: false, streaming: true, progress: 0.7 },
    { loading: false, streaming: false },
  ];
  
  for (const s of loadingStates) {
    const acc = accessibleLoadingState(s.loading, s.streaming, (s as any).progress);
    console.log(`    ${acc.braille}  "${acc.ariaLabel}"`);
  }
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 5: Semantic Fingerprints
// ═══════════════════════════════════════════════════════════════════

function demo5_fingerprints() {
  separator('8-Dot Braille: Semantic Fingerprints');

  console.log('  Each text gets a fixed-width Braille "fingerprint":');
  console.log('  Similar texts → similar patterns → recognizable by touch\n');

  const texts = [
    'The capital of France is Paris.',
    'The capital of France is Paris!',
    'Paris is the capital of France.',
    'The weather today is sunny and warm.',
    'Machine learning uses neural networks.',
  ];

  const fingerprints: string[] = [];
  for (const text of texts) {
    const fp = semanticFingerprint(text, 8);
    fingerprints.push(fp);
    console.log(`    ${fp}  "${text.slice(0, 45)}${text.length > 45 ? '...' : ''}"`);
  }

  console.log('\n  Similarity matrix (Braille fingerprint comparison):');
  console.log('    ' + ''.padStart(6) + texts.map((_, i) => `[${i}]`.padStart(6)).join(''));
  for (let i = 0; i < fingerprints.length; i++) {
    let row = `    [${i}]`;
    for (let j = 0; j < fingerprints.length; j++) {
      const sim = fingerprintSimilarity(fingerprints[i], fingerprints[j]);
      row += `${(sim * 100).toFixed(0).padStart(5)}%`;
    }
    console.log(row);
  }

  console.log('\n  Use case: A blind user can "feel" if two AI responses are');
  console.log('  similar by comparing their Braille fingerprints on a display.');
}

// ═══════════════════════════════════════════════════════════════════
// DEMO 6: Reactive Braille Signal with Live AI
// ═══════════════════════════════════════════════════════════════════

async function demo6_reactiveBraille() {
  separator('8-Dot Braille: Reactive Signal + AI Inference');

  console.log('  A Braille signal that reactively updates as AI state changes:\n');

  // Use mock provider for demo
  setDefaultProvider(createMockProvider({
    response: '{"answer": "42", "confidence": 0.95}',
    delay: 500,
    chunkSize: 5,
  }));

  const synapse = createSynapse({
    model: 'mock',
    signature: 'question -> answer, confidence',
    dependencies: () => ({ question: 'What is the meaning of life?' }),
    autoTrigger: false,
    stream: true,
    debounce: 0,
  });

  // Create a reactive Braille signal from the synapse state
  const brailleState = createBrailleSignal(synapse.state);

  // Watch it update
  const log: string[] = [];
  const dispose = createEffect(() => {
    const state = brailleState();
    log.push(`  ${state.braille}  "${state.ariaLabel}"`);
  });

  // Trigger and wait
  console.log('  Timeline of Braille state changes:');
  console.log('  (Each line = a reactive update)\n');
  
  await synapse.trigger();
  await new Promise(r => setTimeout(r, 100));

  for (const entry of log) {
    console.log(entry);
  }

  dispose();
  synapse.dispose();
}

// ═══════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║     Synapse.js — 8-Dot Braille: AI + Accessibility Native       ║
║                                                                  ║
║  "What if AI state was something you could TOUCH?"              ║
║                                                                  ║
║  8-dot Braille encodes 256 patterns per character.              ║
║  Screen readers speak them. Braille displays render them.       ║
║  Sighted users see data visualization. Everyone gets signal.    ║
╚══════════════════════════════════════════════════════════════════╝`);

  demo1_encoding();
  demo2_confidence();
  demo3_graphState();
  await demo4_streaming();
  demo5_fingerprints();
  await demo6_reactiveBraille();

  separator('BRAILLE DEMO COMPLETE');
  console.log('  8-dot Braille in Synapse.js enables:');
  console.log('');
  console.log('    • Confidence levels as tactile/visual bars');
  console.log('    • Reactive graph state as a touchable dashboard');
  console.log('    • Streaming indicators that work on Braille displays');
  console.log('    • Semantic fingerprints for output comparison by touch');
  console.log('    • Full ARIA integration for screen readers');
  console.log('    • Zero-overhead encoding (pure Unicode, no dependencies)');
  console.log('');
  console.log('  The same data is SIMULTANEOUSLY:');
  console.log('    → Visual (dot patterns create data viz for sighted users)');
  console.log('    → Tactile (Braille displays render them physically)');
  console.log('    → Semantic (screen readers announce meaningful labels)');
  console.log('    → Compact (256 symbols = information-dense encoding)');
  console.log('');
  console.log('  This is what "accessible-native" means: not an afterthought,');
  console.log('  but a fundamental encoding that serves ALL modalities.');
  console.log('');
}

main().catch(console.error);
