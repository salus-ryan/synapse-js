/**
 * Layer 1: Braille-Native Tokenizer
 * 
 * A byte-level tokenizer where the vocabulary IS the 256 Unicode Braille
 * characters (U+2800–U+28FF). Each byte maps 1:1 to a Braille cell.
 * 
 * This enables models that "think" in Braille natively — no text-to-Braille
 * conversion needed. The raw token stream IS the Braille stream.
 * 
 * Compatible with byte-level models (ByT5, MambaByte, byte-level GPT).
 */

// ─────────────────────────────────────────────────────────────────
// Core Constants
// ─────────────────────────────────────────────────────────────────

/** Unicode Braille Patterns block start (U+2800) */
const BRAILLE_BASE = 0x2800;

/** Total vocabulary size: 256 Braille characters (8-dot = 2^8) */
export const VOCAB_SIZE = 256;

/** Special token IDs within the 256-char space */
export const SPECIAL_TOKENS = {
  PAD: 0x00,    // ⠀ (empty cell = padding)
  BOS: 0x01,    // ⠁ (dot 1 = beginning of sequence)
  EOS: 0x02,    // ⠂ (dot 2 = end of sequence)
  UNK: 0x03,    // ⠃ (dots 1+2 = unknown)
  SEP: 0x04,    // ⠄ (dot 3 = separator)
  MASK: 0xFF,   // ⣿ (all dots = mask token)
} as const;

// ─────────────────────────────────────────────────────────────────
// Braille Tokenizer
// ─────────────────────────────────────────────────────────────────

export interface BrailleToken {
  /** Token ID (0-255) */
  id: number;
  /** The Braille character */
  braille: string;
  /** Original byte value */
  byte: number;
  /** Human-readable ASCII equivalent (if printable) */
  ascii: string | null;
}

export interface TokenizedSequence {
  /** Token IDs (each 0-255) */
  ids: number[];
  /** Braille string representation */
  braille: string;
  /** Original byte length */
  byteLength: number;
  /** Includes BOS/EOS */
  hasSpecialTokens: boolean;
}

/**
 * The Braille-Native Tokenizer.
 * 
 * Maps bytes ↔ Braille characters 1:1.
 * Zero information loss. Zero ambiguity. O(1) per token.
 * 
 * This is deliberately the simplest possible tokenizer —
 * the novelty is in WHAT it tokenizes into, not HOW.
 */
export class BrailleTokenizer {
  private byteToChar: string[];
  private charToByte: Map<string, number>;

  constructor() {
    // Build bidirectional lookup tables
    this.byteToChar = new Array(256);
    this.charToByte = new Map();

    for (let i = 0; i < 256; i++) {
      const brailleChar = String.fromCodePoint(BRAILLE_BASE + i);
      this.byteToChar[i] = brailleChar;
      this.charToByte.set(brailleChar, i);
    }
  }

  /** 
   * Encode a string to Braille token IDs.
   * Each UTF-8 byte becomes one token (0-255).
   */
  encode(text: string, addSpecialTokens = true): TokenizedSequence {
    const bytes = new TextEncoder().encode(text);
    const ids: number[] = [];

    if (addSpecialTokens) {
      ids.push(SPECIAL_TOKENS.BOS);
    }

    for (const byte of bytes) {
      ids.push(byte);
    }

    if (addSpecialTokens) {
      ids.push(SPECIAL_TOKENS.EOS);
    }

    return {
      ids,
      braille: ids.map(id => this.byteToChar[id]).join(''),
      byteLength: bytes.length,
      hasSpecialTokens: addSpecialTokens,
    };
  }

  /**
   * Decode Braille token IDs back to text.
   * Strips special tokens, converts bytes back to UTF-8.
   */
  decode(ids: number[], stripSpecialTokens = true): string {
    let tokenIds = ids;

    if (stripSpecialTokens) {
      tokenIds = ids.filter(id => 
        id !== SPECIAL_TOKENS.BOS && 
        id !== SPECIAL_TOKENS.EOS && 
        id !== SPECIAL_TOKENS.PAD &&
        id !== SPECIAL_TOKENS.MASK
      );
    }

    const bytes = new Uint8Array(tokenIds);
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  }

  /**
   * Convert a Braille string directly to token IDs.
   * Each Braille character becomes its byte offset from U+2800.
   */
  brailleToIds(braille: string): number[] {
    const ids: number[] = [];
    for (const char of braille) {
      const byte = this.charToByte.get(char);
      if (byte !== undefined) {
        ids.push(byte);
      } else {
        ids.push(SPECIAL_TOKENS.UNK);
      }
    }
    return ids;
  }

  /**
   * Convert token IDs to a Braille string.
   * Each ID (0-255) becomes its corresponding Braille character.
   */
  idsToBraille(ids: number[]): string {
    return ids.map(id => this.byteToChar[id & 0xFF]).join('');
  }

  /**
   * Get a single token's full info.
   */
  getToken(id: number): BrailleToken {
    const byte = id & 0xFF;
    const ascii = byte >= 32 && byte < 127 ? String.fromCharCode(byte) : null;
    return {
      id: byte,
      braille: this.byteToChar[byte],
      byte,
      ascii,
    };
  }

  /**
   * Tokenize with full metadata per token.
   */
  tokenize(text: string): BrailleToken[] {
    const { ids } = this.encode(text, false);
    return ids.map(id => this.getToken(id));
  }

  /** Vocabulary size (always 256 for 8-dot Braille) */
  get vocabSize(): number {
    return VOCAB_SIZE;
  }

  /**
   * Display the full vocabulary as a reference table.
   */
  vocabularyTable(): string {
    const lines: string[] = ['Braille-Native Vocabulary (256 tokens):', ''];
    for (let row = 0; row < 16; row++) {
      const cells: string[] = [];
      for (let col = 0; col < 16; col++) {
        const id = row * 16 + col;
        cells.push(`${this.byteToChar[id]} `);
      }
      lines.push(`  ${row.toString(16).toUpperCase()}x: ${cells.join('')}`);
    }
    return lines.join('\n');
  }

  /**
   * Compute compression ratio vs standard BPE tokenizer.
   * Byte-level has 1:1 token:byte ratio (no subword merges).
   * For comparison, GPT-4 averages ~3.5 bytes per token.
   */
  compressionAnalysis(text: string): {
    brailleTokens: number;
    estimatedBPETokens: number;
    bytesPerBrailleToken: number;
    bytesPerBPEToken: number;
    brailleOverhead: number;
  } {
    const bytes = new TextEncoder().encode(text).length;
    const brailleTokens = bytes; // 1:1
    const estimatedBPETokens = Math.ceil(bytes / 3.5); // GPT-4 average
    
    return {
      brailleTokens,
      estimatedBPETokens,
      bytesPerBrailleToken: 1,
      bytesPerBPEToken: 3.5,
      brailleOverhead: brailleTokens / estimatedBPETokens,
    };
  }
}

// ─────────────────────────────────────────────────────────────────
// Embedding Compatibility Layer
// ─────────────────────────────────────────────────────────────────

/**
 * A minimal embedding table for the 256-token Braille vocabulary.
 * In a real implementation, these would be learned during training.
 * Here we provide a structured initialization based on Braille geometry.
 * 
 * The key insight: Braille dot positions have spatial meaning.
 * We encode this geometry into the initial embeddings so the model
 * starts with structural awareness of the Braille space.
 */
export function createBrailleEmbeddings(dim: number): Float32Array[] {
  const embeddings: Float32Array[] = [];

  for (let id = 0; id < 256; id++) {
    const embedding = new Float32Array(dim);
    
    // Extract individual dot states (8 dots)
    const dots = [
      (id >> 0) & 1, // dot 1 (top-left)
      (id >> 1) & 1, // dot 2 (mid-left)
      (id >> 2) & 1, // dot 3 (bottom-left)
      (id >> 3) & 1, // dot 4 (top-right)
      (id >> 4) & 1, // dot 5 (mid-right)
      (id >> 5) & 1, // dot 6 (bottom-right)
      (id >> 6) & 1, // dot 7 (lower-left)
      (id >> 7) & 1, // dot 8 (lower-right)
    ];

    // Geometric encoding: first 8 dims directly encode dot states
    for (let d = 0; d < 8 && d < dim; d++) {
      embedding[d] = dots[d] * 2 - 1; // -1 or +1
    }

    // Density encoding: dim 8 = number of raised dots / 8
    if (dim > 8) {
      embedding[8] = dots.reduce((a, b) => a + b, 0) / 8;
    }

    // Symmetry encoding: dim 9 = left-right balance
    if (dim > 9) {
      const leftSum = dots[0] + dots[1] + dots[2] + dots[6];
      const rightSum = dots[3] + dots[4] + dots[5] + dots[7];
      embedding[9] = (leftSum - rightSum) / 4;
    }

    // Vertical position encoding: dim 10 = top-heavy vs bottom-heavy
    if (dim > 10) {
      const topSum = dots[0] + dots[3];
      const bottomSum = dots[2] + dots[5] + dots[6] + dots[7];
      embedding[10] = (topSum - bottomSum) / 4;
    }

    // Byte value encoding: sinusoidal position (like transformer PE)
    for (let d = 11; d < dim; d++) {
      const freq = 1.0 / Math.pow(10000, (d - 11) / (dim - 11));
      embedding[d] = d % 2 === 0 ? Math.sin(id * freq) : Math.cos(id * freq);
    }

    embeddings.push(embedding);
  }

  return embeddings;
}

// ─────────────────────────────────────────────────────────────────
// Singleton instance
// ─────────────────────────────────────────────────────────────────

export const brailleTokenizer = new BrailleTokenizer();
