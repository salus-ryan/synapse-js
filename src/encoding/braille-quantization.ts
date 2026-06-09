/**
 * Layer 2: Braille Quantization
 * 
 * A novel weight quantization scheme where model weights are stored
 * as 8-dot Braille characters. Each weight = 1 Braille cell = 8 bits.
 * 
 * This creates the first quantization format that is simultaneously:
 * - Machine-efficient (INT8-equivalent, 256 discrete levels)
 * - Human-readable (visual dot patterns show weight distribution)
 * - Human-touchable (native on refreshable Braille displays)
 * 
 * A quantized model is literally a Unicode text document.
 */

// ─────────────────────────────────────────────────────────────────
// Core Constants
// ─────────────────────────────────────────────────────────────────

const BRAILLE_BASE = 0x2800;

/**
 * Dot layout in 8-dot Braille cell:
 * 
 *   ┌───┬───┐
 *   │ 1 │ 4 │  ← Row 0 (MSB)
 *   ├───┼───┤
 *   │ 2 │ 5 │  ← Row 1
 *   ├───┼───┤
 *   │ 3 │ 6 │  ← Row 2
 *   ├───┼───┤
 *   │ 7 │ 8 │  ← Row 3 (LSB)
 *   └───┴───┘
 * 
 * Bit mapping (standard Braille encoding):
 *   Dot 1 = bit 0, Dot 2 = bit 1, Dot 3 = bit 2
 *   Dot 4 = bit 3, Dot 5 = bit 4, Dot 6 = bit 5
 *   Dot 7 = bit 6, Dot 8 = bit 7
 * 
 * For quantization, we reinterpret as:
 *   Bits 7-6 (dots 8,7): Sign + overflow
 *   Bits 5-3 (dots 6,5,4): Exponent / scale
 *   Bits 2-0 (dots 3,2,1): Mantissa
 */

// ─────────────────────────────────────────────────────────────────
// Quantization Formats
// ─────────────────────────────────────────────────────────────────

export type QuantFormat = 'linear' | 'logarithmic' | 'symmetric' | 'geometric';

export interface QuantConfig {
  /** Quantization format */
  format: QuantFormat;
  /** Number of bits (always 8 for Braille, but allows sub-byte modes) */
  bits: 8 | 4 | 2;
  /** Whether to use per-channel or per-tensor scaling */
  granularity: 'per_tensor' | 'per_channel' | 'per_group';
  /** Group size for per-group quantization */
  groupSize?: number;
  /** Whether zero-point is included */
  symmetric: boolean;
}

export interface QuantizedTensor {
  /** Braille-encoded weights */
  braille: string;
  /** Scale factor(s) for dequantization */
  scale: number | number[];
  /** Zero point(s) */
  zeroPoint: number | number[];
  /** Original shape */
  shape: number[];
  /** Quantization config used */
  config: QuantConfig;
  /** Statistics */
  stats: TensorStats;
}

export interface TensorStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  sparsity: number; // fraction of near-zero weights
  brailleEntropy: number; // information density in Braille representation
}

// ─────────────────────────────────────────────────────────────────
// Core Quantization Functions
// ─────────────────────────────────────────────────────────────────

/**
 * Quantize a float32 value to a Braille character.
 */
export function quantizeScalar(
  value: number, 
  scale: number, 
  zeroPoint: number,
  format: QuantFormat = 'linear'
): string {
  let quantized: number;

  switch (format) {
    case 'linear':
      quantized = Math.round(value / scale + zeroPoint);
      break;
    case 'logarithmic':
      // Log-scale quantization (better for weights with heavy tails)
      const sign = value >= 0 ? 1 : -1;
      const logVal = Math.log1p(Math.abs(value) / scale);
      quantized = Math.round(sign * logVal * 127 + 128);
      break;
    case 'symmetric':
      // Symmetric around zero (no zero-point)
      quantized = Math.round(value / scale * 127 + 128);
      break;
    case 'geometric':
      // Dot density proportional to magnitude
      const magnitude = Math.abs(value) / scale;
      const dotCount = Math.round(magnitude * 8);
      quantized = densityToByte(Math.min(dotCount, 8));
      break;
  }

  // Clamp to [0, 255]
  quantized = Math.max(0, Math.min(255, quantized!));
  return String.fromCodePoint(BRAILLE_BASE + quantized);
}

/**
 * Dequantize a Braille character back to float32.
 */
export function dequantizeScalar(
  brailleChar: string,
  scale: number,
  zeroPoint: number,
  format: QuantFormat = 'linear'
): number {
  const byte = brailleChar.codePointAt(0)! - BRAILLE_BASE;

  switch (format) {
    case 'linear':
      return (byte - zeroPoint) * scale;
    case 'logarithmic':
      const centered = (byte - 128) / 127;
      const sign = centered >= 0 ? 1 : -1;
      return sign * (Math.expm1(Math.abs(centered)) * scale);
    case 'symmetric':
      return ((byte - 128) / 127) * scale;
    case 'geometric':
      const dots = byteToDensity(byte);
      return (dots / 8) * scale;
  }
}

/**
 * Quantize an entire tensor to Braille.
 */
export function quantizeTensor(
  data: Float32Array | number[],
  shape: number[],
  config: QuantConfig = { format: 'symmetric', bits: 8, granularity: 'per_tensor', symmetric: true }
): QuantizedTensor {
  const values = data instanceof Float32Array ? data : new Float32Array(data);
  
  // Compute statistics
  let min = Infinity, max = -Infinity, sum = 0, sumSq = 0;
  let nearZero = 0;
  
  for (const v of values) {
    min = Math.min(min, v);
    max = Math.max(max, v);
    sum += v;
    sumSq += v * v;
    if (Math.abs(v) < 1e-6) nearZero++;
  }
  
  const mean = sum / values.length;
  const std = Math.sqrt(sumSq / values.length - mean * mean);
  const sparsity = nearZero / values.length;

  // Compute scale and zero-point
  let scale: number;
  let zeroPoint: number;

  if (config.symmetric) {
    const absMax = Math.max(Math.abs(min), Math.abs(max));
    scale = absMax / 127;
    zeroPoint = 128;
  } else {
    scale = (max - min) / 255;
    zeroPoint = Math.round(-min / scale);
  }

  if (scale === 0) scale = 1; // Avoid division by zero

  // Quantize each value to Braille
  let braille = '';
  for (const v of values) {
    braille += quantizeScalar(v, scale, zeroPoint, config.format);
  }

  // Compute Braille entropy (information density)
  const charCounts = new Map<string, number>();
  for (const ch of braille) {
    charCounts.set(ch, (charCounts.get(ch) || 0) + 1);
  }
  let entropy = 0;
  for (const count of charCounts.values()) {
    const p = count / braille.length;
    if (p > 0) entropy -= p * Math.log2(p);
  }

  return {
    braille,
    scale,
    zeroPoint,
    shape,
    config,
    stats: { min, max, mean, std, sparsity, brailleEntropy: entropy },
  };
}

/**
 * Dequantize a Braille tensor back to float32.
 */
export function dequantizeTensor(quantized: QuantizedTensor): Float32Array {
  const scale = typeof quantized.scale === 'number' ? quantized.scale : quantized.scale[0];
  const zeroPoint = typeof quantized.zeroPoint === 'number' ? quantized.zeroPoint : quantized.zeroPoint[0];
  const values = new Float32Array(quantized.braille.length);

  let i = 0;
  for (const ch of quantized.braille) {
    values[i++] = dequantizeScalar(ch, scale, zeroPoint, quantized.config.format);
  }

  return values;
}

// ─────────────────────────────────────────────────────────────────
// Tactile Analysis Tools
// ─────────────────────────────────────────────────────────────────

/**
 * Compute tactile density map — how "heavy" each region of the tensor feels.
 * Returns dot count per Braille character (0-8 dots raised).
 */
export function tactileDensityMap(quantized: QuantizedTensor): number[] {
  const densities: number[] = [];
  for (const ch of quantized.braille) {
    const byte = ch.codePointAt(0)! - BRAILLE_BASE;
    densities.push(popcount(byte));
  }
  return densities;
}

/**
 * Generate a tactile histogram — distribution of dot densities.
 * Useful for understanding weight distribution by touch.
 */
export function tactileHistogram(quantized: QuantizedTensor): {
  histogram: number[];
  visualization: string;
  description: string;
} {
  const densities = tactileDensityMap(quantized);
  const histogram = new Array(9).fill(0); // 0-8 dots

  for (const d of densities) {
    histogram[d]++;
  }

  // Visualize as Braille bar chart
  const maxCount = Math.max(...histogram);
  const barWidth = 20;
  const lines: string[] = [];
  
  for (let dots = 0; dots <= 8; dots++) {
    const barLen = Math.round((histogram[dots] / maxCount) * barWidth);
    const bar = String.fromCodePoint(BRAILLE_BASE + densityToByte(dots)).repeat(barLen);
    const pct = ((histogram[dots] / densities.length) * 100).toFixed(1);
    lines.push(`  ${dots} dots: ${bar.padEnd(barWidth)} ${pct}%`);
  }

  // Describe the distribution
  const avgDensity = densities.reduce((a, b) => a + b, 0) / densities.length;
  let description: string;
  if (avgDensity < 2) description = 'Sparse (mostly empty cells — highly prunable)';
  else if (avgDensity < 4) description = 'Light (weights concentrated near zero)';
  else if (avgDensity < 5) description = 'Balanced (uniform weight distribution)';
  else if (avgDensity < 7) description = 'Dense (heavy weights, high magnitude)';
  else description = 'Saturated (weights at maximum — potential overflow)';

  return {
    histogram,
    visualization: lines.join('\n'),
    description: `${description} (avg ${avgDensity.toFixed(1)} dots/cell)`,
  };
}

/**
 * Layer-wise tactile summary — feel the shape of a neural network.
 */
export function modelTactileSummary(layers: QuantizedTensor[]): string {
  const lines: string[] = ['Model Tactile Summary:', ''];

  layers.forEach((layer, i) => {
    const densities = tactileDensityMap(layer);
    const avgDensity = densities.reduce((a, b) => a + b, 0) / densities.length;
    const bar = String.fromCodePoint(BRAILLE_BASE + densityToByte(Math.round(avgDensity)));
    const sparsityPct = (layer.stats.sparsity * 100).toFixed(0);
    
    lines.push(
      `  Layer ${String(i).padStart(2)}: ${bar.repeat(8)} ` +
      `density=${avgDensity.toFixed(1)} sparsity=${sparsityPct}% ` +
      `entropy=${layer.stats.brailleEntropy.toFixed(2)}bits`
    );
  });

  return lines.join('\n');
}

// ─────────────────────────────────────────────────────────────────
// Quantization Error Analysis
// ─────────────────────────────────────────────────────────────────

/**
 * Compute quantization error metrics.
 */
export function quantizationError(
  original: Float32Array | number[],
  quantized: QuantizedTensor
): {
  mse: number;
  mae: number;
  maxError: number;
  snr: number;
  brailleEfficiency: number;
} {
  const reconstructed = dequantizeTensor(quantized);
  const orig = original instanceof Float32Array ? original : new Float32Array(original);

  let mse = 0, mae = 0, maxError = 0, signalPower = 0;

  for (let i = 0; i < orig.length; i++) {
    const error = Math.abs(orig[i] - reconstructed[i]);
    mse += error * error;
    mae += error;
    maxError = Math.max(maxError, error);
    signalPower += orig[i] * orig[i];
  }

  mse /= orig.length;
  mae /= orig.length;
  signalPower /= orig.length;

  const snr = 10 * Math.log10(signalPower / mse); // Signal-to-noise ratio in dB
  
  // Braille efficiency: how much of the 8-bit space is actually used
  const uniqueChars = new Set(quantized.braille).size;
  const brailleEfficiency = uniqueChars / 256;

  return { mse, mae, maxError, snr, brailleEfficiency };
}

// ─────────────────────────────────────────────────────────────────
// Model Serialization (Braille Document Format)
// ─────────────────────────────────────────────────────────────────

export interface BrailleModelDocument {
  /** Model metadata */
  header: {
    name: string;
    layers: number;
    totalParams: number;
    format: QuantFormat;
    createdAt: string;
  };
  /** Quantized layers */
  layers: Array<{
    name: string;
    shape: number[];
    braille: string;
    scale: number;
    zeroPoint: number;
  }>;
  /** The entire model as a single Braille string (for Braille display) */
  fullBraille: string;
}

/**
 * Serialize quantized layers into a Braille Model Document.
 * The resulting document is a readable Unicode text file.
 */
export function createBrailleModelDocument(
  name: string,
  layers: Array<{ name: string; tensor: QuantizedTensor }>
): BrailleModelDocument {
  const totalParams = layers.reduce((sum, l) => sum + l.tensor.braille.length, 0);
  
  return {
    header: {
      name,
      layers: layers.length,
      totalParams,
      format: layers[0]?.tensor.config.format || 'symmetric',
      createdAt: new Date().toISOString(),
    },
    layers: layers.map(l => ({
      name: l.name,
      shape: l.tensor.shape,
      braille: l.tensor.braille,
      scale: typeof l.tensor.scale === 'number' ? l.tensor.scale : l.tensor.scale[0],
      zeroPoint: typeof l.tensor.zeroPoint === 'number' ? l.tensor.zeroPoint : l.tensor.zeroPoint[0],
    })),
    fullBraille: layers.map(l => l.tensor.braille).join(''),
  };
}

/**
 * Format a Braille Model Document for display on a Braille terminal.
 * Wraps to specified width (standard Braille display = 40 or 80 cells).
 */
export function formatForBrailleDisplay(
  doc: BrailleModelDocument, 
  width: number = 40
): string {
  const lines: string[] = [];
  
  lines.push(`⠶ ${doc.header.name} ⠶`);
  lines.push(`⠤ ${doc.header.totalParams} params ⠤`);
  lines.push('');

  for (const layer of doc.layers) {
    lines.push(`⠶ ${layer.name} [${layer.shape.join('×')}] ⠶`);
    
    // Wrap Braille content to display width
    for (let i = 0; i < layer.braille.length; i += width) {
      lines.push(layer.braille.slice(i, i + width));
    }
    lines.push('');
  }

  return lines.join('\n');
}

// ─────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────

/** Count set bits (population count) */
function popcount(n: number): number {
  let count = 0;
  while (n) {
    count += n & 1;
    n >>= 1;
  }
  return count;
}

/** 
 * Convert a dot density (0-8) to a representative byte.
 * Uses a pattern that distributes dots evenly.
 */
function densityToByte(dots: number): number {
  // Progressive fill pattern (fills from outside in)
  const patterns = [
    0b00000000, // 0 dots
    0b00000001, // 1 dot  (top-left)
    0b00001001, // 2 dots (top corners)
    0b00101001, // 3 dots
    0b00101101, // 4 dots
    0b01101101, // 5 dots
    0b01111101, // 6 dots
    0b01111111, // 7 dots
    0b11111111, // 8 dots (full)
  ];
  return patterns[Math.min(Math.max(dots, 0), 8)];
}

/**
 * Convert a byte to its dot density (number of bits set).
 */
function byteToDensity(byte: number): number {
  return popcount(byte);
}
