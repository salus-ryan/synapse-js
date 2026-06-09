/**
 * Tool Primitive — Reactive Function Calling
 * 
 * Tools are functions that AI models can invoke. In Synapse.js,
 * tools are reactive: their results feed back into the computation
 * graph, potentially triggering new inferences.
 */

import { createSignal, ReadonlySignal } from '../core/reactive.js';

// --- Types ---

export interface ToolDefinition<TInput = any, TOutput = any> {
  /** Tool name (used by the model to invoke it) */
  name: string;
  /** Human-readable description */
  description: string;
  /** JSON Schema for the input parameters */
  parameters: Record<string, any>;
  /** The implementation function */
  execute: (input: TInput) => Promise<TOutput> | TOutput;
}

export interface ToolInstance<TInput = any, TOutput = any> {
  /** The tool definition (for passing to models) */
  definition: ToolDefinition<TInput, TOutput>;
  /** Last result from this tool (reactive) */
  lastResult: ReadonlySignal<TOutput | undefined>;
  /** Whether the tool is currently executing */
  loading: ReadonlySignal<boolean>;
  /** Invoke the tool manually */
  invoke: (input: TInput) => Promise<TOutput>;
  /** Invocation count */
  callCount: ReadonlySignal<number>;
}

// --- Create Tool ---

/**
 * Creates a reactive tool that can be used by Synapses.
 * 
 * @example
 * const weatherTool = createTool({
 *   name: 'get_weather',
 *   description: 'Get current weather for a location',
 *   parameters: {
 *     type: 'object',
 *     properties: {
 *       location: { type: 'string', description: 'City name' },
 *     },
 *     required: ['location'],
 *   },
 *   execute: async ({ location }) => {
 *     const res = await fetch(`https://api.weather.com/${location}`);
 *     return res.json();
 *   },
 * });
 */
export function createTool<TInput = any, TOutput = any>(
  definition: ToolDefinition<TInput, TOutput>
): ToolInstance<TInput, TOutput> {
  const [lastResult, setLastResult] = createSignal<TOutput | undefined>(undefined);
  const [loading, setLoading] = createSignal(false);
  const [callCount, setCallCount] = createSignal(0);

  async function invoke(input: TInput): Promise<TOutput> {
    setLoading(true);
    try {
      const result = await definition.execute(input);
      setLastResult(result as any);
      setCallCount(prev => prev + 1);
      return result;
    } finally {
      setLoading(false);
    }
  }

  return {
    definition,
    lastResult: lastResult as ReadonlySignal<TOutput | undefined>,
    loading: loading as ReadonlySignal<boolean>,
    invoke,
    callCount: callCount as ReadonlySignal<number>,
  };
}

// --- Tool Registry ---

export class ToolRegistry {
  private tools = new Map<string, ToolInstance>();

  register<TInput, TOutput>(tool: ToolInstance<TInput, TOutput>): void {
    this.tools.set(tool.definition.name, tool as any);
  }

  get(name: string): ToolInstance | undefined {
    return this.tools.get(name);
  }

  getAll(): ToolInstance[] {
    return [...this.tools.values()];
  }

  /** Get tool definitions in OpenAI function-calling format */
  toOpenAITools(): any[] {
    return this.getAll().map(tool => ({
      type: 'function',
      function: {
        name: tool.definition.name,
        description: tool.definition.description,
        parameters: tool.definition.parameters,
      },
    }));
  }

  /** Execute a tool call from a model response */
  async execute(name: string, args: any): Promise<any> {
    const tool = this.tools.get(name);
    if (!tool) {
      throw new Error(`Tool not found: ${name}`);
    }
    return tool.invoke(args);
  }
}

/** Global tool registry */
export const globalToolRegistry = new ToolRegistry();
