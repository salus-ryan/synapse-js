/**
 * Experiment 6: Hallucination Detection via Braille Confidence Channel
 * 
 * Demonstrates that the Braille streaming codec can detect hallucination
 * by monitoring per-token confidence (dot density) in real time.
 * 
 * Strategy: Ask the model both factual and hallucination-inducing prompts,
 * capture logprobs/confidence, encode as Braille, and show that confidence
 * drops ("confidence craters") correlate with hallucinated content.
 * 
 * Requires: Ollama running locally with qwen2.5:0.5b
 */

import {
  createSignal,
  createSynapse,
  createOpenAIProvider,
  setDefaultProvider,
  BrailleStreamEncoder,
  brailleTokenizer,
  semanticFingerprint,
  fingerprintSimilarity,
} from '../src/index';

const MODEL = process.env.MODEL || 'qwen2.5:0.5b';
const BASE_URL = process.env.OLLAMA_URL || 'http://localhost:11434/v1';

const provider = createOpenAIProvider({ apiKey: 'ollama', baseURL: BASE_URL });
setDefaultProvider(provider);

function ts(): string {
  return new Date().toISOString();
}

function header(title: string) {
  console.log(`\n${'═'.repeat(70)}`);
  console.log(`  [${ts()}] ${title}`);
  console.log(`${'═'.repeat(70)}\n`);
}

// ─────────────────────────────────────────────────────────────────
// Helper: Run inference and capture streaming chunks with timing
// ─────────────────────────────────────────────────────────────────

interface StreamResult {
  text: string;
  chunks: string[];
  chunkTimes: number[];
  totalTime: number;
}

async function streamPrompt(prompt: string, maxTokens: number = 50): Promise<StreamResult> {
  const chunks: string[] = [];
  const chunkTimes: number[] = [];
  const start = Date.now();

  const response = await fetch(`${BASE_URL}/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ollama' },
    body: JSON.stringify({
      model: MODEL,
      messages: [{ role: 'user', content: prompt }],
      temperature: 0.7,
      max_tokens: maxTokens,
      stream: true,
    }),
  });

  if (!response.ok) throw new Error(`API error: ${response.status}`);

  const reader = response.body?.getReader();
  if (!reader) throw new Error('No response body');

  const decoder = new TextDecoder();
  let accumulated = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value, { stream: true });
    for (const line of text.split('\n')) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;
        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta?.content;
          if (delta) {
            chunks.push(delta);
            chunkTimes.push(Date.now() - start);
            accumulated += delta;
          }
        } catch {}
      }
    }
  }

  reader.releaseLock();
  return { text: accumulated, chunks, chunkTimes, totalTime: Date.now() - start };
}

// ─────────────────────────────────────────────────────────────────
// Helper: Encode streaming result into Braille with simulated confidence
// We use inter-token timing as a proxy for confidence:
// - Fast tokens = high confidence (model is certain, generates quickly)
// - Slow tokens = low confidence (model is uncertain, takes longer)
// ─────────────────────────────────────────────────────────────────

interface BrailleAnalysis {
  contentBraille: string;
  confidenceBraille: string;
  confidenceValues: number[];
  avgConfidence: number;
  minConfidence: number;
  confidenceVariance: number;
  craterCount: number; // number of sudden confidence drops
  craterDepth: number; // average depth of craters
}

function analyzeWithBraille(result: StreamResult): BrailleAnalysis {
  const encoder = new BrailleStreamEncoder({ modelId: MODEL, maxTokens: 100 });

  // Compute inter-token delays (proxy for confidence)
  // Normalize: fastest = 1.0 confidence, slowest = 0.0
  const delays: number[] = [];
  for (let i = 1; i < result.chunkTimes.length; i++) {
    delays.push(result.chunkTimes[i] - result.chunkTimes[i - 1]);
  }
  if (delays.length === 0) delays.push(0);

  // First token has no delay reference, assume medium confidence
  const maxDelay = Math.max(...delays, 1);
  const minDelay = Math.min(...delays);

  const confidenceValues: number[] = [0.8]; // first token baseline
  for (const delay of delays) {
    // Inverse relationship: shorter delay = higher confidence
    const conf = 1.0 - (delay - minDelay) / (maxDelay - minDelay + 1);
    confidenceValues.push(Math.max(0.1, Math.min(1.0, conf)));
  }

  // Encode each chunk
  for (let i = 0; i < result.chunks.length; i++) {
    encoder.encodeToken(result.chunks[i], confidenceValues[i] || 0.5);
  }

  const stream = encoder.finalize();

  // Detect confidence craters (drops of >0.3 from local average)
  let craterCount = 0;
  let craterDepthSum = 0;
  const windowSize = 3;

  for (let i = windowSize; i < confidenceValues.length; i++) {
    const windowAvg = confidenceValues.slice(i - windowSize, i).reduce((a, b) => a + b, 0) / windowSize;
    const drop = windowAvg - confidenceValues[i];
    if (drop > 0.3) {
      craterCount++;
      craterDepthSum += drop;
    }
  }

  const avgConfidence = confidenceValues.reduce((a, b) => a + b, 0) / confidenceValues.length;
  const variance = confidenceValues.reduce((sum, v) => sum + (v - avgConfidence) ** 2, 0) / confidenceValues.length;

  return {
    contentBraille: stream.contentBraille,
    confidenceBraille: stream.confidenceBraille,
    confidenceValues,
    avgConfidence,
    minConfidence: Math.min(...confidenceValues),
    confidenceVariance: variance,
    craterCount,
    craterDepth: craterCount > 0 ? craterDepthSum / craterCount : 0,
  };
}

// ─────────────────────────────────────────────────────────────────
// Experiment 6a: Factual vs Hallucination-Inducing Prompts
// ─────────────────────────────────────────────────────────────────

async function experiment6a() {
  header('6a — Confidence Patterns: Factual vs Hallucination-Inducing');

  const prompts = [
    {
      label: 'FACTUAL (known answer)',
      prompt: 'What is the capital of France? One word answer.',
      expectedConfidence: 'high',
    },
    {
      label: 'FACTUAL (arithmetic)',
      prompt: 'What is 7 * 8? Just the number.',
      expectedConfidence: 'high',
    },
    {
      label: 'HALLUCINATION-PRONE (fake entity)',
      prompt: 'Describe the scientific contributions of Dr. Reginald Thornberry III to quantum linguistics in the 1940s.',
      expectedConfidence: 'low/variable',
    },
    {
      label: 'HALLUCINATION-PRONE (impossible specifics)',
      prompt: 'What was the exact temperature in downtown Tokyo at 3:47 PM on March 14, 1987?',
      expectedConfidence: 'low/variable',
    },
    {
      label: 'HALLUCINATION-PRONE (fake reference)',
      prompt: 'Summarize the findings of the 2019 paper "Neural Braille Encoding for Tactile Transformers" by Chen et al.',
      expectedConfidence: 'low/variable',
    },
  ];

  const results: Array<{ label: string; analysis: BrailleAnalysis; text: string }> = [];

  // Warmup
  console.log('  Warming up model...');
  await streamPrompt('hi', 5).catch(() => {});
  console.log('  ✓ Ready\n');

  for (const p of prompts) {
    console.log(`  ─── ${p.label} ───`);
    console.log(`  Prompt: "${p.prompt.slice(0, 60)}${p.prompt.length > 60 ? '...' : ''}"`);

    const result = await streamPrompt(p.prompt, 40);
    const analysis = analyzeWithBraille(result);

    console.log(`  Response: "${result.text.slice(0, 60).trim()}${result.text.length > 60 ? '...' : ''}"`);
    console.log(`  Content:    ${analysis.contentBraille.slice(0, 40)}`);
    console.log(`  Confidence: ${analysis.confidenceBraille.slice(0, 40)}`);
    console.log(`  Avg conf:   ${(analysis.avgConfidence * 100).toFixed(1)}%  Min: ${(analysis.minConfidence * 100).toFixed(1)}%  Variance: ${analysis.confidenceVariance.toFixed(4)}`);
    console.log(`  Craters:    ${analysis.craterCount} (avg depth: ${(analysis.craterDepth * 100).toFixed(1)}%)`);
    console.log(`  Expected:   ${p.expectedConfidence}\n`);

    results.push({ label: p.label, analysis, text: result.text });
  }

  // Summary comparison
  console.log(`\n  ┌──────────────────────────────────────────────────────────────────┐`);
  console.log(`  │ Prompt Type              │ Avg Conf │ Variance │ Craters │ Min   │`);
  console.log(`  ├──────────────────────────────────────────────────────────────────┤`);

  for (const r of results) {
    const label = r.label.slice(0, 24).padEnd(24);
    const avg = (r.analysis.avgConfidence * 100).toFixed(1).padStart(6) + '%';
    const vari = r.analysis.confidenceVariance.toFixed(4).padStart(8);
    const craters = String(r.analysis.craterCount).padStart(7);
    const min = (r.analysis.minConfidence * 100).toFixed(1).padStart(5) + '%';
    console.log(`  │ ${label} │ ${avg} │ ${vari} │ ${craters} │ ${min} │`);
  }
  console.log(`  └──────────────────────────────────────────────────────────────────┘`);

  // Statistical comparison: factual vs hallucination-prone
  const factual = results.filter(r => r.label.startsWith('FACTUAL'));
  const halluc = results.filter(r => r.label.startsWith('HALLUC'));

  const avgFactualConf = factual.reduce((s, r) => s + r.analysis.avgConfidence, 0) / factual.length;
  const avgHallucConf = halluc.reduce((s, r) => s + r.analysis.avgConfidence, 0) / halluc.length;
  const avgFactualVar = factual.reduce((s, r) => s + r.analysis.confidenceVariance, 0) / factual.length;
  const avgHallucVar = halluc.reduce((s, r) => s + r.analysis.confidenceVariance, 0) / halluc.length;
  const avgFactualCraters = factual.reduce((s, r) => s + r.analysis.craterCount, 0) / factual.length;
  const avgHallucCraters = halluc.reduce((s, r) => s + r.analysis.craterCount, 0) / halluc.length;

  console.log(`\n  [${ts()}] Aggregate comparison:`);
  console.log(`    Factual prompts:      avg_conf=${(avgFactualConf * 100).toFixed(1)}%  variance=${avgFactualVar.toFixed(4)}  craters=${avgFactualCraters.toFixed(1)}`);
  console.log(`    Hallucination-prone:  avg_conf=${(avgHallucConf * 100).toFixed(1)}%  variance=${avgHallucVar.toFixed(4)}  craters=${avgHallucCraters.toFixed(1)}`);
  console.log(`    Confidence gap:       ${((avgFactualConf - avgHallucConf) * 100).toFixed(1)} percentage points`);
  console.log(`    Variance ratio:       ${(avgHallucVar / (avgFactualVar || 0.0001)).toFixed(2)}x higher for hallucination`);

  return { avgFactualConf, avgHallucConf, avgFactualVar, avgHallucVar, avgFactualCraters, avgHallucCraters };
}

// ─────────────────────────────────────────────────────────────────
// Experiment 6b: Multi-Model Confidence Divergence
// ─────────────────────────────────────────────────────────────────

async function experiment6b() {
  header('6b — Multi-Model Confidence Divergence on Hallucination');

  // Run same hallucination-prone prompt multiple times (simulating multiple models)
  // and measure how confidence tracks correlate
  const prompt = 'What year was the novel "The Quantum Garden" by Derek Künsken published? Just the year.';

  console.log(`  Prompt: "${prompt}"`);
  console.log(`  Running 4 independent generations (simulating multi-model braid)...\n`);

  const runs: StreamResult[] = [];
  for (let i = 0; i < 4; i++) {
    const result = await streamPrompt(prompt, 20);
    runs.push(result);
    console.log(`    Run ${i + 1}: "${result.text.trim()}" (${result.chunks.length} chunks, ${result.totalTime}ms)`);
  }

  // Analyze each
  const analyses = runs.map(r => analyzeWithBraille(r));

  // Compare pairwise confidence correlation
  console.log(`\n  Pairwise confidence analysis:`);
  let totalCorrelation = 0;
  let pairs = 0;

  for (let i = 0; i < analyses.length; i++) {
    for (let j = i + 1; j < analyses.length; j++) {
      const minLen = Math.min(analyses[i].confidenceValues.length, analyses[j].confidenceValues.length);
      const a = analyses[i].confidenceValues.slice(0, minLen);
      const b = analyses[j].confidenceValues.slice(0, minLen);

      // Pearson correlation
      const meanA = a.reduce((s, v) => s + v, 0) / a.length;
      const meanB = b.reduce((s, v) => s + v, 0) / b.length;
      let num = 0, denA = 0, denB = 0;
      for (let k = 0; k < minLen; k++) {
        num += (a[k] - meanA) * (b[k] - meanB);
        denA += (a[k] - meanA) ** 2;
        denB += (b[k] - meanB) ** 2;
      }
      const corr = (denA > 0 && denB > 0) ? num / (Math.sqrt(denA) * Math.sqrt(denB)) : 0;
      totalCorrelation += corr;
      pairs++;

      // Text agreement
      const textMatch = runs[i].text.trim() === runs[j].text.trim();
      console.log(`    Run ${i + 1} vs Run ${j + 1}: conf_corr=${corr.toFixed(3)}  text_match=${textMatch ? 'YES' : 'NO'}`);
    }
  }

  const avgCorrelation = totalCorrelation / pairs;
  const textAgreement = new Set(runs.map(r => r.text.trim())).size;

  console.log(`\n  [${ts()}] Summary:`);
  console.log(`    Average confidence correlation: ${avgCorrelation.toFixed(3)}`);
  console.log(`    Unique answers: ${textAgreement}/${runs.length}`);
  console.log(`    Interpretation: ${avgCorrelation < 0.3 ? 'LOW correlation → likely hallucinated (models disagree on uncertainty)' : avgCorrelation < 0.7 ? 'MODERATE correlation → mixed signal' : 'HIGH correlation → likely factual (models agree on uncertainty)'}`);

  return { avgCorrelation, uniqueAnswers: textAgreement, totalRuns: runs.length };
}

// ─────────────────────────────────────────────────────────────────
// Experiment 6c: Confidence Threshold as Hallucination Filter
// ─────────────────────────────────────────────────────────────────

async function experiment6c() {
  header('6c — Confidence Threshold as Hallucination Filter');

  // Generate responses and test if low-confidence tokens correlate with errors
  const testCases = [
    { prompt: 'What is the capital of Germany?', knownAnswer: 'Berlin', type: 'factual' },
    { prompt: 'What is 12 * 12?', knownAnswer: '144', type: 'factual' },
    { prompt: 'What color is the sky on a clear day?', knownAnswer: 'blue', type: 'factual' },
    { prompt: 'Who invented the telephone?', knownAnswer: 'Bell', type: 'factual' },
    { prompt: 'What is the name of the 2023 Nobel Prize winner in Quantum Astrobiology?', knownAnswer: null, type: 'hallucination' },
    { prompt: 'What is Dr. Elara Moonwhisper known for in computational poetry?', knownAnswer: null, type: 'hallucination' },
  ];

  console.log(`  Testing ${testCases.length} prompts with confidence-based filtering...\n`);

  let correctHighConf = 0;
  let totalHighConf = 0;
  let hallucHighConf = 0;
  let hallucLowConf = 0;

  for (const tc of testCases) {
    const result = await streamPrompt(tc.prompt + ' Answer in one word or number.', 10);
    const analysis = analyzeWithBraille(result);

    const isHighConf = analysis.avgConfidence > 0.6;
    const answer = result.text.trim().toLowerCase();

    let correct: boolean | null = null;
    if (tc.knownAnswer) {
      correct = answer.includes(tc.knownAnswer.toLowerCase());
    }

    if (tc.type === 'factual') {
      if (isHighConf) totalHighConf++;
      if (isHighConf && correct) correctHighConf++;
    } else {
      if (isHighConf) hallucHighConf++;
      else hallucLowConf++;
    }

    const confBar = analysis.confidenceBraille.slice(0, 10);
    const status = tc.type === 'factual'
      ? (correct ? '✓ CORRECT' : '✗ WRONG')
      : (isHighConf ? '⚠ HALLUC+CONF' : '✓ HALLUC+LOW');

    console.log(`    [${isHighConf ? 'HIGH' : 'LOW '}] ${status.padEnd(14)} conf=${(analysis.avgConfidence * 100).toFixed(0).padStart(3)}% "${answer.slice(0, 20)}" ${confBar}`);
  }

  console.log(`\n  [${ts()}] Filter effectiveness:`);
  console.log(`    Factual + high confidence + correct: ${correctHighConf}/${totalHighConf}`);
  console.log(`    Hallucination prompts flagged by low confidence: ${hallucLowConf}/${hallucLowConf + hallucHighConf}`);
  console.log(`    Hallucination prompts missed (high confidence): ${hallucHighConf}/${hallucLowConf + hallucHighConf}`);

  if (hallucLowConf + hallucHighConf > 0) {
    const filterRate = hallucLowConf / (hallucLowConf + hallucHighConf);
    console.log(`    Filter catch rate: ${(filterRate * 100).toFixed(0)}%`);
  }

  return { correctHighConf, totalHighConf, hallucHighConf, hallucLowConf };
}

// ─────────────────────────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────────────────────────

async function main() {
  console.log(`
╔══════════════════════════════════════════════════════════════════════╗
║   Braille-Native AI: Hallucination Detection Experiment             ║
║                                                                      ║
║   Testing confidence channel as hallucination signal                  ║
║   Model: ${MODEL.padEnd(20)}                                        ║
║   Timestamp: ${new Date().toISOString()}                    ║
╚══════════════════════════════════════════════════════════════════════╝`);

  const r6a = await experiment6a();
  const r6b = await experiment6b();
  const r6c = await experiment6c();

  header('SUMMARY — Hallucination Experiment Results');

  console.log(`  6a. Confidence gap (factual vs hallucination):`);
  console.log(`      Factual avg:       ${(r6a.avgFactualConf * 100).toFixed(1)}%`);
  console.log(`      Hallucination avg: ${(r6a.avgHallucConf * 100).toFixed(1)}%`);
  console.log(`      Gap:               ${((r6a.avgFactualConf - r6a.avgHallucConf) * 100).toFixed(1)} pp`);
  console.log(`      Variance ratio:    ${(r6a.avgHallucVar / (r6a.avgFactualVar || 0.0001)).toFixed(2)}x\n`);

  console.log(`  6b. Multi-run confidence correlation:`);
  console.log(`      Avg correlation:   ${r6b.avgCorrelation.toFixed(3)}`);
  console.log(`      Unique answers:    ${r6b.uniqueAnswers}/${r6b.totalRuns}\n`);

  console.log(`  6c. Confidence threshold filter:`);
  console.log(`      Factual correct (high conf): ${r6c.correctHighConf}/${r6c.totalHighConf}`);
  console.log(`      Hallucination caught (low conf): ${r6c.hallucLowConf}/${r6c.hallucLowConf + r6c.hallucHighConf}`);
  console.log('');
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
