/**
 * Synapse Primitive — The Reactive AI Inference Node
 * 
 * A Synapse wraps an LLM call in a reactive computation graph.
 * When its dependencies (input signals) change, it automatically
 * re-invokes the model, streams tokens, and propagates results
 * to downstream nodes.
 * 
 * Key innovations:
 * - Automatic dependency tracking (like Solid.js computed)
 * - Built-in debouncing to prevent rapid re-fires
 * - Cancellation of in-flight requests when inputs change
 * - Token streaming as reactive updates
 * - Signature-based prompt compilation (inspired by DSPy)
 */

import {
  createSignal,
  createEffect,
  createComputed,
  ReadonlySignal,
  AsyncSignalState,
  createAsyncSignal,
  batch,
} from '../core/reactive.js';
import { ModelProvider, InferenceRequest, InferenceResponse } from '../runtime/providers.js';

// --- Types ---

export interface SynapseSignature {
  /** DSPy-style signature: "input1, input2 -> output1, output2" */
  signature: string;
  /** Parsed input field names */
  inputs: string[];
  /** Parsed output field names */
  outputs: string[];
}

export interface SynapseConfig<TInput extends Record<string, any>, TOutput> {
  /** The model identifier (e.g., 'gpt-4o', 'claude-3.5-sonnet') */
  model: string;
  /** DSPy-style signature: "context, query -> answer, confidence" */
  signature: string;
  /** System prompt (optional, can be auto-generated from signature) */
  system?: string;
  /** Reactive dependency function that returns the current inputs */
  dependencies: () => TInput;
  /** Custom provider (optional, uses default if not specified) */
  provider?: ModelProvider;
  /** Debounce time in ms before triggering inference (default: 300) */
  debounce?: number;
  /** Temperature for the model (default: 0.7) */
  temperature?: number;
  /** Max tokens for the response */
  maxTokens?: number;
  /** Whether to stream the response (default: true) */
  stream?: boolean;
  /** Custom output parser */
  parse?: (raw: string) => TOutput;
  /** Whether to auto-trigger on dependency changes (default: true) */
  autoTrigger?: boolean;
}

export interface SynapseInstance<TOutput> {
  /** Read the current output (reactive) */
  output: ReadonlySignal<TOutput | undefined>;
  /** Read the full async state (loading, streaming, error) */
  state: ReadonlySignal<AsyncSignalState<TOutput>>;
  /** The raw text stream (reactive, updates token-by-token) */
  stream: ReadonlySignal<string>;
  /** Manually trigger inference */
  trigger: () => Promise<TOutput>;
  /** Cancel in-flight inference */
  cancel: () => void;
  /** Dispose this synapse and all subscriptions */
  dispose: () => void;
  /** The parsed signature */
  signature: SynapseSignature;
}

// --- Signature Parser ---

export function parseSignature(sig: string): SynapseSignature {
  const [inputPart, outputPart] = sig.split('->').map(s => s.trim());
  if (!inputPart || !outputPart) {
    throw new Error(`Invalid signature: "${sig}". Expected format: "input1, input2 -> output1, output2"`);
  }
  const inputs = inputPart.split(',').map(s => s.trim()).filter(Boolean);
  const outputs = outputPart.split(',').map(s => s.trim()).filter(Boolean);
  return { signature: sig, inputs, outputs };
}

// --- Prompt Compiler ---

function compilePrompt(
  signature: SynapseSignature,
  inputs: Record<string, any>,
  systemOverride?: string
): { system: string; user: string } {
  const system = systemOverride || 
    `You are a precise AI that takes the following inputs and produces structured outputs.\n` +
    `Your task signature: ${signature.signature}\n` +
    `You MUST respond with a JSON object containing exactly these fields: ${signature.outputs.join(', ')}.\n` +
    `Respond ONLY with valid JSON, no markdown fences, no explanation.`;

  const inputLines = signature.inputs.map(name => {
    const value = inputs[name];
    const serialized = typeof value === 'string' ? value : JSON.stringify(value);
    return `[${name}]: ${serialized}`;
  });

  const user = inputLines.join('\n');

  return { system, user };
}

// --- Default Output Parser ---

function defaultParse<TOutput>(raw: string, signature: SynapseSignature): TOutput {
  // Try JSON parse first
  try {
    // Strip markdown fences if present
    const cleaned = raw.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
    const parsed = JSON.parse(cleaned);
    return parsed as TOutput;
  } catch {
    // If single output field, return raw text as that field
    if (signature.outputs.length === 1) {
      return { [signature.outputs[0]]: raw.trim() } as TOutput;
    }
    // Otherwise return raw
    return raw as unknown as TOutput;
  }
}

// --- Global Provider Registry ---

let defaultProvider: ModelProvider | null = null;

export function setDefaultProvider(provider: ModelProvider): void {
  defaultProvider = provider;
}

export function getDefaultProvider(): ModelProvider | null {
  return defaultProvider;
}

// --- Create Synapse ---

/**
 * Creates a reactive AI inference node.
 * 
 * @example
 * const analyzer = createSynapse({
 *   model: 'gpt-4o',
 *   signature: 'text -> sentiment, confidence',
 *   dependencies: () => ({ text: userInput() }),
 * });
 * 
 * // Reading the output is reactive — it triggers re-evaluation
 * createEffect(() => {
 *   console.log("Sentiment:", analyzer.output()?.sentiment);
 * });
 */
export function createSynapse<
  TInput extends Record<string, any>,
  TOutput = Record<string, any>
>(config: SynapseConfig<TInput, TOutput>): SynapseInstance<TOutput> {
  const signature = parseSignature(config.signature);
  const debounceMs = config.debounce ?? 300;
  const shouldStream = config.stream ?? true;
  const autoTrigger = config.autoTrigger ?? true;

  // Internal state
  const asyncState = createAsyncSignal<TOutput>();
  const [streamText, setStreamText] = createSignal('');
  let abortController: AbortController | null = null;
  let debounceTimer: ReturnType<typeof setTimeout> | null = null;
  let disposed = false;

  // The output signal (convenience accessor)
  const output = createComputed(() => asyncState.state().value);

  // Core inference function
  async function runInference(): Promise<TOutput> {
    // Cancel any in-flight request
    cancel();

    const provider = config.provider || defaultProvider;
    if (!provider) {
      throw new Error(
        'No model provider configured. Call setDefaultProvider() or pass a provider in config.'
      );
    }

    // Get current inputs
    const inputs = config.dependencies();

    // Compile the prompt from signature + inputs
    const { system, user } = compilePrompt(signature, inputs, config.system);

    // Prepare the request
    const request: InferenceRequest = {
      model: config.model,
      messages: [
        { role: 'system', content: system },
        { role: 'user', content: user },
      ],
      temperature: config.temperature ?? 0.7,
      maxTokens: config.maxTokens,
      stream: shouldStream,
    };

    abortController = new AbortController();
    const signal = abortController.signal;

    batch(() => {
      asyncState.startLoading();
      setStreamText('');
    });

    try {
      if (shouldStream) {
        asyncState.startStreaming();
        let accumulated = '';

        await provider.stream(request, signal, (chunk: string) => {
          if (signal.aborted) return;
          accumulated += chunk;
          batch(() => {
            setStreamText(accumulated);
            asyncState.stream(
              accumulated as unknown as TOutput,
              () => accumulated as unknown as TOutput
            );
          });
        });

        // Parse final result
        const parsed = config.parse 
          ? config.parse(accumulated) 
          : defaultParse<TOutput>(accumulated, signature);
        
        asyncState.resolve(parsed);
        return parsed;
      } else {
        const response = await provider.complete(request, signal);
        const parsed = config.parse 
          ? config.parse(response.content) 
          : defaultParse<TOutput>(response.content, signature);
        
        batch(() => {
          setStreamText(response.content);
          asyncState.resolve(parsed);
        });
        return parsed;
      }
    } catch (error: any) {
      if (error.name === 'AbortError') {
        // Cancelled — not an error
        return output.peek() as TOutput;
      }
      asyncState.reject(error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally {
      abortController = null;
    }
  }

  // Debounced trigger
  function debouncedTrigger(): void {
    if (debounceTimer) clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      if (!disposed) {
        runInference().catch(() => {}); // Errors handled via state
      }
    }, debounceMs);
  }

  // Auto-trigger effect: watches dependencies and re-fires
  let disposeEffect: (() => void) | null = null;
  if (autoTrigger) {
    disposeEffect = createEffect(() => {
      // Reading dependencies triggers tracking
      config.dependencies();
      // Debounced re-inference
      debouncedTrigger();
    });
  }

  function cancel(): void {
    if (abortController) {
      abortController.abort();
      abortController = null;
    }
    if (debounceTimer) {
      clearTimeout(debounceTimer);
      debounceTimer = null;
    }
  }

  function dispose(): void {
    disposed = true;
    cancel();
    if (disposeEffect) disposeEffect();
  }

  return {
    output,
    state: asyncState.state,
    stream: streamText as ReadonlySignal<string>,
    trigger: runInference,
    cancel,
    dispose,
    signature,
  };
}
