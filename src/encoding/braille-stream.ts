/**
 * Layer 3: Streaming Braille Codec
 * 
 * A real-time token-to-Braille transformation protocol that encodes
 * not just text content but also confidence, position, and model state
 * into parallel Braille channels.
 * 
 * On a Braille display, a user simultaneously feels:
 * - WHAT the model is saying (text encoded as Braille)
 * - HOW SURE it is (dot density = token probability)
 * - WHERE it is in generation (tactile progress indicator)
 * - WHICH model produced it (model signature pattern)
 * 
 * This is the first streaming protocol designed for tactile consumption.
 */

const BRAILLE_BASE = 0x2800;

// ─────────────────────────────────────────────────────────────────
// Frame Structure
// ─────────────────────────────────────────────────────────────────

/**
 * A single Braille Stream Frame.
 * 
 * Each frame represents one token's worth of information
 * encoded as multiple Braille cells (channels).
 * 
 * Physical layout on a Braille display (per token):
 *   [content] [confidence] [position] [model_id]
 *   4 cells    1 cell       1 cell     1 cell    = 7 cells per token
 * 
 * Or in compact mode:
 *   [content+confidence_merged]
 *   1 cell per byte (confidence encoded in dot pattern selection)
 */
export interface BrailleFrame {
  /** Frame sequence number */
  seq: number;
  /** ISO 8601 timestamp */
  timestamp: string;
  /** Content channel: the actual text byte(s) as Braille */
  content: string;
  /** Confidence channel: single Braille char (dot density = probability) */
  confidence: string;
  /** Position channel: progress through generation (0-255 mapped to Braille) */
  position: string;
  /** Model ID channel: identifies which model/strand produced this */
  modelId: string;
  /** Raw token probability (0-1) */
  probability: number;
  /** Original text content */
  text: string;
  /** Compact single-line representation */
  compact: string;
}

/**
 * A complete Braille Stream — accumulated frames from one generation.
 */
export interface BrailleStream {
  /** All frames in order */
  frames: BrailleFrame[];
  /** Full content as Braille string */
  contentBraille: string;
  /** Full confidence track as Braille string */
  confidenceBraille: string;
  /** Decoded text */
  text: string;
  /** Stream metadata */
  meta: StreamMeta;
}

export interface StreamMeta {
  modelId: string;
  startTime: string;
  endTime: string | null;
  totalTokens: number;
  avgConfidence: number;
  tokensPerSecond: number;
  isComplete: boolean;
}

// ─────────────────────────────────────────────────────────────────
// Braille Stream Encoder
// ─────────────────────────────────────────────────────────────────

export class BrailleStreamEncoder {
  private frames: BrailleFrame[] = [];
  private startTime: number;
  private modelId: string;
  private modelSignature: string;
  private maxTokens: number;

  constructor(config: {
    modelId: string;
    maxTokens?: number;
  }) {
    this.modelId = config.modelId;
    this.maxTokens = config.maxTokens || 1024;
    this.startTime = Date.now();
    // Generate a stable 1-char Braille signature for this model
    this.modelSignature = this.computeModelSignature(config.modelId);
  }

  /**
   * Encode a new token into the stream.
   */
  encodeToken(text: string, probability: number = 1.0): BrailleFrame {
    const seq = this.frames.length;
    const timestamp = new Date().toISOString();

    // Content channel: text bytes as Braille
    const textBytes = new TextEncoder().encode(text);
    const content = Array.from(textBytes)
      .map(b => String.fromCodePoint(BRAILLE_BASE + b))
      .join('');

    // Confidence channel: probability mapped to dot density
    const confidence = this.probabilityToBraille(probability);

    // Position channel: progress through max_tokens
    const progress = Math.min(seq / this.maxTokens, 1);
    const position = String.fromCodePoint(BRAILLE_BASE + Math.round(progress * 255));

    // Model ID channel
    const modelIdChar = this.modelSignature;

    // Compact: merge content with confidence as a single readable line
    const compact = content + confidence;

    const frame: BrailleFrame = {
      seq,
      timestamp,
      content,
      confidence,
      position,
      modelId: modelIdChar,
      probability,
      text,
      compact,
    };

    this.frames.push(frame);
    return frame;
  }

  /**
   * Encode a chunk of text (multiple characters) with uniform confidence.
   */
  encodeChunk(text: string, probability: number = 1.0): BrailleFrame[] {
    // Split into individual characters for per-char frames
    const chars = [...text];
    return chars.map(ch => this.encodeToken(ch, probability));
  }

  /**
   * Encode a token with logprob data (from OpenAI-style API).
   */
  encodeWithLogprob(text: string, logprob: number): BrailleFrame {
    const probability = Math.exp(logprob); // logprob → probability
    return this.encodeToken(text, probability);
  }

  /**
   * Finalize the stream and return the complete BrailleStream.
   */
  finalize(): BrailleStream {
    const endTime = Date.now();
    const totalTokens = this.frames.length;
    const elapsed = (endTime - this.startTime) / 1000;

    const avgConfidence = totalTokens > 0
      ? this.frames.reduce((sum, f) => sum + f.probability, 0) / totalTokens
      : 0;

    return {
      frames: this.frames,
      contentBraille: this.frames.map(f => f.content).join(''),
      confidenceBraille: this.frames.map(f => f.confidence).join(''),
      text: this.frames.map(f => f.text).join(''),
      meta: {
        modelId: this.modelId,
        startTime: new Date(this.startTime).toISOString(),
        endTime: new Date(endTime).toISOString(),
        totalTokens,
        avgConfidence,
        tokensPerSecond: elapsed > 0 ? totalTokens / elapsed : 0,
        isComplete: true,
      },
    };
  }

  /**
   * Get the current stream state (before finalization).
   */
  currentState(): BrailleStream {
    const totalTokens = this.frames.length;
    const elapsed = (Date.now() - this.startTime) / 1000;
    const avgConfidence = totalTokens > 0
      ? this.frames.reduce((sum, f) => sum + f.probability, 0) / totalTokens
      : 0;

    return {
      frames: this.frames,
      contentBraille: this.frames.map(f => f.content).join(''),
      confidenceBraille: this.frames.map(f => f.confidence).join(''),
      text: this.frames.map(f => f.text).join(''),
      meta: {
        modelId: this.modelId,
        startTime: new Date(this.startTime).toISOString(),
        endTime: null,
        totalTokens,
        avgConfidence,
        tokensPerSecond: elapsed > 0 ? totalTokens / elapsed : 0,
        isComplete: false,
      },
    };
  }

  // ─────────────────────────────────────────────────────────────
  // Private helpers
  // ─────────────────────────────────────────────────────────────

  private probabilityToBraille(p: number): string {
    // Map probability to dot count (0-8 dots)
    // High confidence = more dots raised = denser cell
    const dotCount = Math.round(p * 8);
    const byte = this.densityPattern(dotCount);
    return String.fromCodePoint(BRAILLE_BASE + byte);
  }

  private densityPattern(dots: number): number {
    // Fill dots from most perceptually salient positions
    // Order: 1,4,2,5,3,6,7,8 (top-down, alternating sides)
    const fillOrder = [0, 3, 1, 4, 2, 5, 6, 7]; // bit positions
    let byte = 0;
    for (let i = 0; i < Math.min(dots, 8); i++) {
      byte |= (1 << fillOrder[i]);
    }
    return byte;
  }

  private computeModelSignature(modelId: string): string {
    // Hash model ID to a stable Braille character
    let hash = 0;
    for (let i = 0; i < modelId.length; i++) {
      hash = ((hash << 5) - hash + modelId.charCodeAt(i)) | 0;
    }
    return String.fromCodePoint(BRAILLE_BASE + (Math.abs(hash) % 256));
  }
}

// ─────────────────────────────────────────────────────────────────
// Multi-Stream Braiding (Multiple Models Streaming Simultaneously)
// ─────────────────────────────────────────────────────────────────

/**
 * Interleaved multi-model Braille stream.
 * Shows all models' outputs side-by-side on a Braille display.
 */
export interface InterleavedStream {
  /** Per-model streams */
  streams: Map<string, BrailleStreamEncoder>;
  /** Interleaved timeline (all frames sorted by timestamp) */
  timeline: Array<BrailleFrame & { source: string }>;
  /** Multi-line display format for Braille terminal */
  display: (width: number) => string;
}

/**
 * Create an interleaved multi-model stream.
 */
export function createInterleavedStream(
  modelIds: string[],
  maxTokens?: number
): InterleavedStream {
  const streams = new Map<string, BrailleStreamEncoder>();
  const timeline: Array<BrailleFrame & { source: string }> = [];

  for (const id of modelIds) {
    streams.set(id, new BrailleStreamEncoder({ modelId: id, maxTokens }));
  }

  return {
    streams,
    timeline,
    display: (width: number) => {
      const lines: string[] = [];
      
      for (const [id, encoder] of streams) {
        const state = encoder.currentState();
        const contentLine = state.contentBraille.slice(-width);
        const confLine = state.confidenceBraille.slice(-width);
        
        lines.push(`${id.slice(0, 12).padEnd(12)} │ ${contentLine}`);
        lines.push(`${''.padEnd(12)} │ ${confLine}`);
        lines.push(`${''.padEnd(12)} │ ${'─'.repeat(Math.min(width, state.meta.totalTokens))}`);
      }

      return lines.join('\n');
    },
  };
}

// ─────────────────────────────────────────────────────────────────
// Display Formatters
// ─────────────────────────────────────────────────────────────────

/**
 * Format a BrailleStream for a standard 40-cell Braille display.
 * Shows content on line 1, confidence on line 2.
 */
export function formatForDisplay(
  stream: BrailleStream,
  width: number = 40
): string[] {
  const lines: string[] = [];
  const content = stream.contentBraille;
  const confidence = stream.confidenceBraille;

  // Header
  lines.push(`⠶ ${stream.meta.modelId.slice(0, width - 4)} ⠶`);

  // Content rows with confidence underline
  for (let i = 0; i < content.length; i += width) {
    lines.push(content.slice(i, i + width));
    lines.push(confidence.slice(i, i + width));
    lines.push(''); // blank line between rows
  }

  // Footer with stats
  const statsLine = `⠤ ${stream.meta.totalTokens}tok ` +
    `${stream.meta.tokensPerSecond.toFixed(1)}t/s ` +
    `conf=${(stream.meta.avgConfidence * 100).toFixed(0)}% ⠤`;
  lines.push(statsLine);

  return lines;
}

/**
 * Format a BrailleStream as an accessible ARIA description.
 */
export function formatAsAria(stream: BrailleStream): string {
  return [
    `Model: ${stream.meta.modelId}`,
    `Status: ${stream.meta.isComplete ? 'Complete' : 'Generating'}`,
    `Tokens: ${stream.meta.totalTokens}`,
    `Speed: ${stream.meta.tokensPerSecond.toFixed(1)} tokens per second`,
    `Average confidence: ${(stream.meta.avgConfidence * 100).toFixed(0)} percent`,
    `Content: ${stream.text}`,
  ].join('. ');
}

// ─────────────────────────────────────────────────────────────────
// Stream Comparison
// ─────────────────────────────────────────────────────────────────

/**
 * Compare two streams — useful for streaming vs non-streaming analysis.
 */
export function compareStreams(a: BrailleStream, b: BrailleStream): {
  textSimilarity: number;
  confidenceCorrelation: number;
  speedRatio: number;
  divergencePoint: number; // token index where they first differ
  report: string;
} {
  // Text similarity (character-level)
  const maxLen = Math.max(a.text.length, b.text.length);
  let matches = 0;
  for (let i = 0; i < Math.min(a.text.length, b.text.length); i++) {
    if (a.text[i] === b.text[i]) matches++;
  }
  const textSimilarity = maxLen > 0 ? matches / maxLen : 1;

  // Find divergence point
  let divergencePoint = -1;
  for (let i = 0; i < Math.min(a.frames.length, b.frames.length); i++) {
    if (a.frames[i].text !== b.frames[i].text) {
      divergencePoint = i;
      break;
    }
  }
  if (divergencePoint === -1) divergencePoint = Math.min(a.frames.length, b.frames.length);

  // Confidence correlation (Pearson)
  const minFrames = Math.min(a.frames.length, b.frames.length);
  let sumA = 0, sumB = 0, sumAB = 0, sumA2 = 0, sumB2 = 0;
  for (let i = 0; i < minFrames; i++) {
    const pa = a.frames[i].probability;
    const pb = b.frames[i].probability;
    sumA += pa;
    sumB += pb;
    sumAB += pa * pb;
    sumA2 += pa * pa;
    sumB2 += pb * pb;
  }
  const n = minFrames || 1;
  const numerator = n * sumAB - sumA * sumB;
  const denominator = Math.sqrt((n * sumA2 - sumA * sumA) * (n * sumB2 - sumB * sumB));
  const confidenceCorrelation = denominator > 0 ? numerator / denominator : 0;

  // Speed comparison
  const speedRatio = b.meta.tokensPerSecond > 0
    ? a.meta.tokensPerSecond / b.meta.tokensPerSecond
    : 1;

  const report = [
    `Stream Comparison: ${a.meta.modelId} vs ${b.meta.modelId}`,
    `  Text similarity: ${(textSimilarity * 100).toFixed(1)}%`,
    `  Confidence correlation: ${confidenceCorrelation.toFixed(3)}`,
    `  Speed ratio: ${speedRatio.toFixed(2)}x`,
    `  Divergence at token: ${divergencePoint}`,
    `  Tokens: ${a.meta.totalTokens} vs ${b.meta.totalTokens}`,
  ].join('\n');

  return { textSimilarity, confidenceCorrelation, speedRatio, divergencePoint, report };
}
