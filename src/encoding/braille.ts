/**
 * 8-Dot Braille Encoding Layer
 * 
 * Makes Synapse.js simultaneously AI-native and accessible-native.
 * 
 * 8-dot Braille (Unicode U+2800–U+28FF) provides 256 unique characters,
 * enabling a dense, tactile-compatible encoding for AI state. This module
 * provides bidirectional encoding between AI inference state and Braille
 * representations that are:
 * 
 * 1. **Screen-reader compatible** — Braille displays render them natively
 * 2. **Information-dense** — 256 symbols encode state in fewer characters
 * 3. **Visual** — Sighted users see pattern-based data visualization
 * 4. **Composable** — Braille signals integrate with the reactive graph
 * 
 * Use cases:
 * - Encode AI confidence levels as tactile patterns
 * - Represent reactive graph topology in Braille for screen readers
 * - Compress streaming tokens into Braille "signal bars"
 * - Provide accessible progress indicators for inference state
 * 
 * 8-dot Braille mapping:
 *   Dot positions: 1 4
 *                  2 5
 *                  3 6
 *                  7 8
 * 
 *   Each dot is a bit: character = U+2800 + (bit pattern)
 *   Bits: d1=0x01, d2=0x02, d3=0x04, d4=0x08, d5=0x10, d6=0x20, d7=0x40, d8=0x80
 */

import { createSignal, createComputed, createEffect, ReadonlySignal } from '../core/reactive.js';

// --- Constants ---

/** Base Unicode codepoint for Braille patterns */
const BRAILLE_BASE = 0x2800;

/** Full block (all 8 dots raised) */
export const BRAILLE_FULL = String.fromCodePoint(BRAILLE_BASE + 0xFF);

/** Empty cell (no dots raised) */
export const BRAILLE_EMPTY = String.fromCodePoint(BRAILLE_BASE);

// --- Core Encoding/Decoding ---

/**
 * Encode a byte (0-255) as a single 8-dot Braille character.
 */
export function byteToBraille(byte: number): string {
  return String.fromCodePoint(BRAILLE_BASE + (byte & 0xFF));
}

/**
 * Decode a Braille character back to a byte (0-255).
 */
export function brailleToByte(char: string): number {
  const cp = char.codePointAt(0);
  if (cp === undefined || cp < BRAILLE_BASE || cp > BRAILLE_BASE + 0xFF) {
    throw new Error(`Invalid Braille character: ${char}`);
  }
  return cp - BRAILLE_BASE;
}

/**
 * Encode a buffer/string into Braille representation.
 * Each byte becomes one Braille character.
 */
export function encodeToBraille(data: Uint8Array | string): string {
  const bytes = typeof data === 'string' 
    ? new TextEncoder().encode(data) 
    : data;
  let result = '';
  for (const byte of bytes) {
    result += byteToBraille(byte);
  }
  return result;
}

/**
 * Decode Braille back to bytes.
 */
export function decodeFromBraille(braille: string): Uint8Array {
  const bytes: number[] = [];
  for (const char of braille) {
    bytes.push(brailleToByte(char));
  }
  return new Uint8Array(bytes);
}

/**
 * Decode Braille back to UTF-8 string.
 */
export function decodeBrailleToString(braille: string): string {
  const bytes = decodeFromBraille(braille);
  return new TextDecoder().decode(bytes);
}

// --- AI State Visualization ---

/**
 * Encode a confidence value (0.0–1.0) as a Braille "bar" pattern.
 * Uses dots bottom-to-top, left column then right column.
 * 
 * 0.0 = ⠀ (empty)
 * 0.125 = ⡀ (dot 7 only — bottom left)
 * 0.25 = ⡠ (dots 7, 6)
 * 0.5 = ⡴ (dots 7, 6, 3, 5)
 * 1.0 = ⣿ (all 8 dots)
 */
export function confidenceToBraille(confidence: number): string {
  const level = Math.round(Math.max(0, Math.min(1, confidence)) * 8);
  // Fill from bottom-up: 7, 3, 2, 1, 8, 6, 5, 4
  const dotOrder = [0x40, 0x04, 0x02, 0x01, 0x80, 0x20, 0x10, 0x08];
  let bits = 0;
  for (let i = 0; i < level; i++) {
    bits |= dotOrder[i];
  }
  return byteToBraille(bits);
}

/**
 * Encode a confidence as a multi-cell horizontal bar.
 * Each cell represents 1/width of the total range.
 */
export function confidenceBar(confidence: number, width: number = 8): string {
  const filled = confidence * width;
  let result = '';
  for (let i = 0; i < width; i++) {
    if (i < Math.floor(filled)) {
      result += BRAILLE_FULL;
    } else if (i < filled) {
      // Partial fill for the last cell
      const partial = filled - Math.floor(filled);
      result += confidenceToBraille(partial);
    } else {
      result += BRAILLE_EMPTY;
    }
  }
  return result;
}

/**
 * Encode streaming progress as a Braille animation frame.
 * Shows a "wave" pattern that moves through the cells.
 */
export function streamingIndicator(frame: number, width: number = 4): string {
  let result = '';
  for (let i = 0; i < width; i++) {
    const phase = ((frame + i) % 8) / 8;
    const bits = Math.round(phase * 255);
    result += byteToBraille(bits);
  }
  return result;
}

// --- Reactive Graph Visualization ---

export interface BrailleNodeState {
  /** Node identifier */
  id: string;
  /** Whether the node is active/computing */
  active: boolean;
  /** Loading state (0-1) */
  progress: number;
  /** Has error */
  error: boolean;
  /** Has value */
  resolved: boolean;
}

/**
 * Encode a reactive graph node's state as a Braille character.
 * 
 * Bit mapping:
 * - Dot 1 (0x01): Has value (resolved)
 * - Dot 2 (0x02): Is active/computing
 * - Dot 3 (0x04): Has error
 * - Dot 4 (0x08): Is streaming
 * - Dots 5-8: Progress level (4 bits = 16 levels)
 */
export function nodeStateToBraille(state: BrailleNodeState): string {
  let bits = 0;
  if (state.resolved) bits |= 0x01;
  if (state.active) bits |= 0x02;
  if (state.error) bits |= 0x04;
  // Progress in upper 4 bits (dots 4-8 mapped)
  const progressBits = Math.round(state.progress * 15) & 0x0F;
  bits |= (progressBits << 4);
  return byteToBraille(bits);
}

/**
 * Encode an array of node states as a Braille string.
 * This provides a tactile "dashboard" of the reactive graph.
 */
export function graphToBraille(nodes: BrailleNodeState[]): string {
  return nodes.map(nodeStateToBraille).join('');
}

// --- Accessible Annotations ---

/**
 * Create an accessible label for a Braille-encoded value.
 * Returns both the Braille representation and an ARIA-compatible description.
 */
export interface AccessibleBraille {
  /** The Braille character(s) */
  braille: string;
  /** Human-readable description for screen readers */
  ariaLabel: string;
  /** Numeric value if applicable */
  value?: number;
}

/**
 * Create an accessible confidence indicator.
 */
export function accessibleConfidence(confidence: number, label?: string): AccessibleBraille {
  const percent = Math.round(confidence * 100);
  return {
    braille: confidenceBar(confidence),
    ariaLabel: `${label ?? 'Confidence'}: ${percent}%`,
    value: confidence,
  };
}

/**
 * Create an accessible loading state indicator.
 */
export function accessibleLoadingState(
  loading: boolean,
  streaming: boolean,
  progress?: number
): AccessibleBraille {
  if (loading && !streaming) {
    return {
      braille: '⠿⠿⠿',
      ariaLabel: 'Loading, please wait',
    };
  }
  if (streaming) {
    const pct = progress ? Math.round(progress * 100) : undefined;
    return {
      braille: confidenceBar(progress ?? 0.5),
      ariaLabel: pct ? `Streaming: ${pct}% complete` : 'Streaming response',
      value: progress,
    };
  }
  return {
    braille: BRAILLE_EMPTY,
    ariaLabel: 'Idle',
  };
}

// --- Reactive Braille Signal ---

/**
 * Creates a reactive Braille signal that encodes an AI signal's state.
 * Updates automatically as the underlying signal changes.
 * 
 * @example
 * const synapse = createSynapse({ ... });
 * const brailleState = createBrailleSignal(synapse.state);
 * 
 * createEffect(() => {
 *   const { braille, ariaLabel } = brailleState();
 *   // Update accessible UI
 *   element.textContent = braille;
 *   element.setAttribute('aria-label', ariaLabel);
 * });
 */
export function createBrailleSignal(
  stateSignal: ReadonlySignal<{ loading: boolean; streaming: boolean; value?: any; error?: any }>
): ReadonlySignal<AccessibleBraille> {
  return createComputed(() => {
    const state = stateSignal();
    if (state.error) {
      return {
        braille: '⣀⣀⣀',
        ariaLabel: `Error: ${state.error.message ?? 'Unknown error'}`,
      };
    }
    if (state.loading) {
      return {
        braille: '⠿⠿⠿',
        ariaLabel: 'Processing inference request',
      };
    }
    if (state.streaming) {
      return {
        braille: '⡇⡇⡇',
        ariaLabel: 'Streaming tokens',
      };
    }
    if (state.value !== undefined) {
      return {
        braille: BRAILLE_FULL + BRAILLE_FULL,
        ariaLabel: 'Response complete',
        value: 1.0,
      };
    }
    return {
      braille: BRAILLE_EMPTY,
      ariaLabel: 'Awaiting input',
    };
  });
}

// --- Semantic Compression (SCL-inspired) ---

/**
 * Compress text into a Braille-encoded semantic hash.
 * This creates a fixed-width "fingerprint" of content that's
 * both visually distinct and tactilely recognizable.
 * 
 * Useful for:
 * - Identifying cached responses by touch
 * - Visual diff of similar outputs
 * - Compact representation in memory graphs
 */
export function semanticFingerprint(text: string, width: number = 8): string {
  // Simple hash-based fingerprint (would use embeddings in production)
  let hash = 0;
  for (let i = 0; i < text.length; i++) {
    hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
  }

  let result = '';
  for (let i = 0; i < width; i++) {
    // Generate deterministic byte from hash + position
    const byte = ((hash >>> (i * 4)) ^ (hash >>> (i * 3 + 1))) & 0xFF;
    result += byteToBraille(byte);
  }
  return result;
}

/**
 * Compare two Braille fingerprints for similarity.
 * Returns 0.0 (completely different) to 1.0 (identical).
 */
export function fingerprintSimilarity(a: string, b: string): number {
  if (a.length !== b.length) return 0;
  let matching = 0;
  let total = 0;
  for (let i = 0; i < a.length; i++) {
    const ba = brailleToByte(a[i]);
    const bb = brailleToByte(b[i]);
    // Count matching bits
    const xor = ba ^ bb;
    for (let bit = 0; bit < 8; bit++) {
      total++;
      if (!(xor & (1 << bit))) matching++;
    }
  }
  return total > 0 ? matching / total : 0;
}
