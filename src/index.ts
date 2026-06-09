/**
 * Synapse.js — An AI-Native Reactive Framework
 * 
 * Synapse.js treats AI inference as a reactive primitive.
 * Like signals are to Solid.js, Synapses are to AI:
 * composable, streamable, and fine-grained reactive nodes
 * in a computation graph.
 * 
 * @packageDocumentation
 */

// Core Reactive Engine
export {
  createSignal,
  createComputed,
  createEffect,
  createAsyncSignal,
  batch,
  type ReadonlySignal,
  type WritableSignal,
  type SignalOptions,
  type AsyncSignalState,
  type Subscriber,
  type Cleanup,
  type Unsubscribe,
} from './core/reactive.js';

// AI Primitives
export {
  createSynapse,
  setDefaultProvider,
  getDefaultProvider,
  parseSignature,
  type SynapseConfig,
  type SynapseInstance,
  type SynapseSignature,
} from './primitives/synapse.js';

export {
  createBraid,
  type BraidConfig,
  type BraidInstance,
  type BraidStrategy,
} from './primitives/braid.js';

export {
  createMemory,
  type MemoryConfig,
  type MemoryInstance,
  type MemoryNode,
} from './primitives/memory.js';

export {
  createTool,
  ToolRegistry,
  globalToolRegistry,
  type ToolDefinition,
  type ToolInstance,
} from './primitives/tool.js';

export {
  createAgent,
  type AgentConfig,
  type AgentInstance,
} from './primitives/agent.js';

export {
  createPipeline,
  type PipelineConfig,
  type PipelineInstance,
  type PipelineStep,
} from './primitives/pipeline.js';

// Runtime / Providers
export {
  createOpenAIProvider,
  createMockProvider,
  type ModelProvider,
  type InferenceRequest,
  type InferenceResponse,
  type Message,
  type OpenAIProviderConfig,
  type MockProviderConfig,
} from './runtime/providers.js';

// Encoding / Accessibility
export {
  byteToBraille,
  brailleToByte,
  encodeToBraille,
  decodeFromBraille,
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
  type AccessibleBraille,
} from './encoding/braille.js';

// Braille-Native Tokenizer (Layer 1)
export {
  BrailleTokenizer,
  brailleTokenizer,
  createBrailleEmbeddings,
  VOCAB_SIZE as BRAILLE_VOCAB_SIZE,
  SPECIAL_TOKENS as BRAILLE_SPECIAL_TOKENS,
  type BrailleToken,
  type TokenizedSequence,
} from './encoding/braille-tokenizer.js';

// Braille Quantization (Layer 2)
export {
  quantizeScalar,
  dequantizeScalar,
  quantizeTensor,
  dequantizeTensor,
  tactileDensityMap,
  tactileHistogram,
  modelTactileSummary,
  quantizationError,
  createBrailleModelDocument,
  formatForBrailleDisplay,
  type QuantFormat,
  type QuantConfig,
  type QuantizedTensor,
  type TensorStats,
  type BrailleModelDocument,
} from './encoding/braille-quantization.js';

// Braille Streaming Codec (Layer 3)
export {
  BrailleStreamEncoder,
  createInterleavedStream,
  formatForDisplay,
  formatAsAria,
  compareStreams,
  type BrailleFrame,
  type BrailleStream,
  type StreamMeta,
  type InterleavedStream,
} from './encoding/braille-stream.js';
