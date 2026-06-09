/**
 * Braid Primitive — Multi-Model Composition
 * 
 * A Braid composes multiple Synapses using declarative strategies:
 * - race: First valid response wins
 * - consensus: Majority agreement among models
 * - fallback: Try each in order until one succeeds
 * - judge: A meta-model evaluates and picks the best response
 * 
 * Braids are reactive: when any constituent Synapse updates,
 * the Braid re-evaluates according to its strategy.
 */

import {
  createSignal,
  createEffect,
  createComputed,
  ReadonlySignal,
  batch,
} from '../core/reactive.js';
import { SynapseInstance } from './synapse.js';

// --- Types ---

export type BraidStrategy = 'race' | 'consensus' | 'fallback' | 'judge';

export interface BraidConfig<TOutput = any> {
  /** How to resolve multiple outputs */
  strategy: BraidStrategy;
  /** Validate an output before accepting it */
  validator?: (output: TOutput) => boolean;
  /** For 'judge' strategy: a function that picks the best output */
  judge?: (outputs: Array<{ output: TOutput; index: number }>) => number;
  /** For 'consensus' strategy: minimum agreement ratio (default: 0.5) */
  consensusThreshold?: number;
  /** For 'consensus' strategy: how to compare two outputs for equality */
  equals?: (a: TOutput, b: TOutput) => boolean;
  /** Timeout in ms for race strategy (default: 30000) */
  timeout?: number;
}

export interface BraidInstance<TOutput = any> {
  /** The resolved output (reactive) */
  output: ReadonlySignal<TOutput | undefined>;
  /** Index of the winning Synapse (reactive) */
  winner: ReadonlySignal<number>;
  /** All collected outputs (reactive) */
  allOutputs: ReadonlySignal<Array<{ output: TOutput | undefined; index: number; valid: boolean }>>;
  /** Whether the braid is still resolving (reactive) */
  resolving: ReadonlySignal<boolean>;
  /** Strategy used */
  strategy: BraidStrategy;
  /** Dispose the braid */
  dispose: () => void;
}

// --- Create Braid ---

/**
 * Creates a multi-model composition node.
 * 
 * @example
 * const braid = createBraid([fastModel, smartModel], {
 *   strategy: 'race',
 *   validator: (output) => output?.confidence > 0.8,
 * });
 * 
 * createEffect(() => {
 *   console.log("Winner:", braid.winner());
 *   console.log("Output:", braid.output());
 * });
 */
export function createBraid<TOutput = any>(
  synapses: SynapseInstance<TOutput>[],
  config: BraidConfig<TOutput>
): BraidInstance<TOutput> {
  const validator = config.validator ?? (() => true);
  const timeout = config.timeout ?? 30000;

  const [output, setOutput] = createSignal<TOutput | undefined>(undefined);
  const [winner, setWinner] = createSignal<number>(-1);
  const [resolving, setResolving] = createSignal(true);
  const [allOutputs, setAllOutputs] = createSignal<
    Array<{ output: TOutput | undefined; index: number; valid: boolean }>
  >([]);

  const disposers: Array<() => void> = [];

  function resolve(value: TOutput, index: number): void {
    if (!resolving.peek()) return; // Already resolved
    batch(() => {
      setOutput(value as any);
      setWinner(index);
      setResolving(false);
    });
  }

  // Strategy implementations
  switch (config.strategy) {
    case 'race':
      implementRace();
      break;
    case 'consensus':
      implementConsensus();
      break;
    case 'fallback':
      implementFallback();
      break;
    case 'judge':
      implementJudge();
      break;
  }

  function implementRace(): void {
    // First valid output wins
    synapses.forEach((synapse, index) => {
      const dispose = createEffect(() => {
        const out = synapse.output();
        if (out !== undefined && validator(out)) {
          resolve(out, index);
        }
      });
      disposers.push(dispose);
    });

    // Timeout
    const timer = setTimeout(() => {
      if (resolving.peek()) {
        // Pick first available even if not valid
        for (let i = 0; i < synapses.length; i++) {
          const out = synapses[i].output.peek();
          if (out !== undefined) {
            resolve(out, i);
            return;
          }
        }
        setResolving(false);
      }
    }, timeout);
    disposers.push(() => clearTimeout(timer));
  }

  function implementConsensus(): void {
    const threshold = config.consensusThreshold ?? 0.5;
    const equals = config.equals ?? ((a, b) => JSON.stringify(a) === JSON.stringify(b));

    const checkConsensus = () => {
      const outputs: Array<{ output: TOutput; index: number }> = [];
      
      synapses.forEach((synapse, index) => {
        const out = synapse.output.peek();
        if (out !== undefined && validator(out)) {
          outputs.push({ output: out, index });
        }
      });

      updateAllOutputs();

      if (outputs.length === 0) return;

      // Group by similarity
      const groups: Array<{ outputs: typeof outputs; count: number }> = [];
      for (const item of outputs) {
        const existingGroup = groups.find(g => equals(g.outputs[0].output, item.output));
        if (existingGroup) {
          existingGroup.outputs.push(item);
          existingGroup.count++;
        } else {
          groups.push({ outputs: [item], count: 1 });
        }
      }

      // Check if any group meets threshold
      const required = Math.ceil(synapses.length * threshold);
      const winning = groups.find(g => g.count >= required);
      if (winning) {
        resolve(winning.outputs[0].output, winning.outputs[0].index);
      }
    };

    synapses.forEach((synapse) => {
      const dispose = createEffect(() => {
        synapse.output();
        checkConsensus();
      });
      disposers.push(dispose);
    });
  }

  function implementFallback(): void {
    // Try in order, use first valid
    let currentIndex = 0;

    const tryNext = () => {
      if (currentIndex >= synapses.length) {
        setResolving(false);
        return;
      }

      const synapse = synapses[currentIndex];
      const dispose = createEffect(() => {
        const out = synapse.output();
        const state = synapse.state();

        if (out !== undefined && validator(out)) {
          resolve(out, currentIndex);
        } else if (state.error) {
          currentIndex++;
          tryNext();
        }
      });
      disposers.push(dispose);
    };

    tryNext();
  }

  function implementJudge(): void {
    if (!config.judge) {
      throw new Error('Judge strategy requires a judge function');
    }

    // Wait for all to complete, then let judge pick
    const checkComplete = () => {
      const outputs: Array<{ output: TOutput; index: number }> = [];
      let allDone = true;

      synapses.forEach((synapse, index) => {
        const state = synapse.state.peek();
        if (state.loading || state.streaming) {
          allDone = false;
        }
        const out = synapse.output.peek();
        if (out !== undefined) {
          outputs.push({ output: out, index });
        }
      });

      updateAllOutputs();

      if (allDone && outputs.length > 0) {
        const winnerIndex = config.judge!(outputs);
        const winningOutput = outputs.find(o => o.index === winnerIndex);
        if (winningOutput) {
          resolve(winningOutput.output, winnerIndex);
        }
      }
    };

    synapses.forEach((synapse) => {
      const dispose = createEffect(() => {
        synapse.state();
        checkComplete();
      });
      disposers.push(dispose);
    });
  }

  function updateAllOutputs(): void {
    const all = synapses.map((synapse, index) => {
      const out = synapse.output.peek();
      return {
        output: out,
        index,
        valid: out !== undefined && validator(out),
      };
    });
    setAllOutputs(all as any);
  }

  function dispose(): void {
    for (const d of disposers) d();
    disposers.length = 0;
  }

  return {
    output: output as ReadonlySignal<TOutput | undefined>,
    winner: winner as ReadonlySignal<number>,
    allOutputs: allOutputs as ReadonlySignal<Array<{ output: TOutput | undefined; index: number; valid: boolean }>>,
    resolving: resolving as ReadonlySignal<boolean>,
    strategy: config.strategy,
    dispose,
  };
}
