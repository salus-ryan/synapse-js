/**
 * Pipeline Primitive — Declarative Inference Chains
 * 
 * A Pipeline connects multiple Synapses in sequence, where the
 * output of one becomes the input of the next. Unlike imperative
 * chaining, pipelines are:
 * 
 * - Declarative: Define the flow, not the execution
 * - Reactive: Changes propagate through the chain
 * - Optimizable: The framework can analyze and optimize the chain
 * - Streamable: Intermediate results stream to downstream nodes
 */

import {
  createSignal,
  createComputed,
  createEffect,
  ReadonlySignal,
  batch,
} from '../core/reactive.js';
import { createSynapse, SynapseInstance, SynapseConfig } from './synapse.js';

// --- Types ---

export interface PipelineStep {
  /** Step name */
  name: string;
  /** Model to use */
  model: string;
  /** Signature for this step */
  signature: string;
  /** System prompt override */
  system?: string;
  /** Temperature */
  temperature?: number;
  /** Transform the input before passing to this step */
  transform?: (input: any) => any;
}

export interface PipelineConfig {
  /** Pipeline steps in order */
  steps: PipelineStep[];
  /** Initial input signal */
  input: () => Record<string, any>;
  /** Whether to auto-trigger (default: true) */
  autoTrigger?: boolean;
  /** Debounce time in ms */
  debounce?: number;
}

export interface PipelineInstance {
  /** Final output of the pipeline (reactive) */
  output: ReadonlySignal<any>;
  /** Intermediate results for each step (reactive) */
  intermediates: ReadonlySignal<Map<string, any>>;
  /** Current step being executed */
  currentStep: ReadonlySignal<string | null>;
  /** Whether the pipeline is running */
  loading: ReadonlySignal<boolean>;
  /** Error if any step failed */
  error: ReadonlySignal<Error | undefined>;
  /** Manually trigger the pipeline */
  trigger: () => Promise<any>;
  /** Dispose the pipeline */
  dispose: () => void;
}

// --- Create Pipeline ---

/**
 * Creates a declarative inference pipeline.
 * 
 * @example
 * const pipeline = createPipeline({
 *   input: () => ({ document: docText() }),
 *   steps: [
 *     {
 *       name: 'extract',
 *       model: 'gpt-4o-mini',
 *       signature: 'document -> entities, relationships',
 *     },
 *     {
 *       name: 'analyze',
 *       model: 'gpt-4o',
 *       signature: 'entities, relationships -> insights, summary',
 *     },
 *     {
 *       name: 'format',
 *       model: 'gpt-4o-mini',
 *       signature: 'insights, summary -> report',
 *     },
 *   ],
 * });
 * 
 * createEffect(() => {
 *   console.log("Current step:", pipeline.currentStep());
 *   console.log("Final output:", pipeline.output());
 * });
 */
export function createPipeline(config: PipelineConfig): PipelineInstance {
  const [output, setOutput] = createSignal<any>(undefined);
  const [intermediates, setIntermediates] = createSignal<Map<string, any>>(new Map());
  const [currentStep, setCurrentStep] = createSignal<string | null>(null);
  const [loading, setLoading] = createSignal(false);
  const [error, setError] = createSignal<Error | undefined>(undefined);

  let disposed = false;

  async function trigger(): Promise<any> {
    if (disposed) return;

    setLoading(true);
    setError(undefined);
    const results = new Map<string, any>();

    try {
      let currentInput = config.input();

      for (const step of config.steps) {
        if (disposed) break;
        setCurrentStep(step.name);

        // Apply transform if provided
        const transformedInput = step.transform 
          ? step.transform(currentInput) 
          : currentInput;

        // Create a one-shot synapse for this step
        const synapse = createSynapse({
          model: step.model,
          signature: step.signature,
          system: step.system,
          dependencies: () => transformedInput,
          autoTrigger: false,
          stream: false,
          temperature: step.temperature,
          debounce: 0,
        });

        const result = await synapse.trigger();
        synapse.dispose();

        results.set(step.name, result);
        setIntermediates(new Map(results));

        // Output of this step becomes input of next
        currentInput = typeof result === 'object' ? result : { result };
      }

      const finalResult = results.get(config.steps[config.steps.length - 1].name);
      setOutput(finalResult);
      setCurrentStep(null);
      return finalResult;
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
      throw err;
    } finally {
      setLoading(false);
    }
  }

  // Auto-trigger effect
  let disposeEffect: (() => void) | null = null;
  if (config.autoTrigger !== false) {
    let debounceTimer: ReturnType<typeof setTimeout> | null = null;
    disposeEffect = createEffect(() => {
      // Track the input (creates reactive dependency)
      config.input();
      
      if (debounceTimer) clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => {
        trigger().catch(() => {});
      }, config.debounce ?? 500);
    });
  }

  function dispose(): void {
    disposed = true;
    if (disposeEffect) disposeEffect();
  }

  return {
    output: output as ReadonlySignal<any>,
    intermediates: intermediates as ReadonlySignal<Map<string, any>>,
    currentStep: currentStep as ReadonlySignal<string | null>,
    loading: loading as ReadonlySignal<boolean>,
    error: error as ReadonlySignal<Error | undefined>,
    trigger,
    dispose,
  };
}
