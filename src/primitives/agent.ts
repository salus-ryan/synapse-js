/**
 * Agent Primitive — Autonomous Reactive Agent
 * 
 * An Agent combines a Synapse, Memory, and Tools into an
 * autonomous reasoning loop. It reactively:
 * - Receives user messages
 * - Decides whether to use tools
 * - Manages its own context/memory
 * - Streams responses back
 * 
 * The Agent is fully reactive: its memory, tool results,
 * and responses are all observable signals.
 */

import {
  createSignal,
  createEffect,
  createComputed,
  ReadonlySignal,
  batch,
} from '../core/reactive.js';
import { createSynapse, SynapseInstance, setDefaultProvider, getDefaultProvider } from './synapse.js';
import { createMemory, MemoryInstance, MemoryConfig } from './memory.js';
import { ToolInstance, ToolRegistry } from './tool.js';
import { ModelProvider, InferenceRequest } from '../runtime/providers.js';

// --- Types ---

export interface AgentConfig {
  /** Agent name */
  name: string;
  /** Model to use */
  model: string;
  /** System prompt */
  system?: string;
  /** Tools available to the agent */
  tools?: ToolInstance[];
  /** Memory configuration */
  memory?: MemoryConfig;
  /** Custom provider */
  provider?: ModelProvider;
  /** Temperature */
  temperature?: number;
  /** Max iterations for tool loops (default: 5) */
  maxIterations?: number;
}

export interface AgentInstance {
  /** The agent's memory */
  memory: MemoryInstance;
  /** Current stream output (reactive) */
  stream: ReadonlySignal<string>;
  /** Whether the agent is processing (reactive) */
  loading: ReadonlySignal<boolean>;
  /** Number of interactions (reactive) */
  interactions: ReadonlySignal<number>;
  /** Send a message and get a response */
  send: (message: string) => Promise<string>;
  /** Reset the agent's memory */
  reset: () => void;
  /** Dispose the agent */
  dispose: () => void;
}

// --- Create Agent ---

/**
 * Creates an autonomous reactive agent.
 * 
 * @example
 * const agent = createAgent({
 *   name: 'Assistant',
 *   model: 'gpt-4o',
 *   system: 'You are a helpful assistant.',
 *   tools: [calculatorTool, weatherTool],
 *   memory: { maxTokens: 8192, strategy: 'sliding_window' },
 * });
 * 
 * const response = await agent.send('What is 2 + 2?');
 * console.log(response);
 */
export function createAgent(config: AgentConfig): AgentInstance {
  const maxIterations = config.maxIterations ?? 5;

  // Create memory
  const memory = createMemory(config.memory ?? {
    maxTokens: 8192,
    strategy: 'sliding_window',
  });

  // Create tool registry
  const registry = new ToolRegistry();
  if (config.tools) {
    for (const tool of config.tools) {
      registry.register(tool);
    }
  }

  // Reactive state
  const [streamText, setStreamText] = createSignal('');
  const [loading, setLoading] = createSignal(false);
  const [interactionCount, setInteractionCount] = createSignal(0);

  // Add system message to memory
  if (config.system) {
    memory.add('system', config.system);
  }

  async function send(message: string): Promise<string> {
    setLoading(true);
    setStreamText('');

    try {
      // Add user message to memory
      memory.add('user', message);

      const provider = config.provider || getDefaultProvider();
      if (!provider) {
        throw new Error('No model provider configured.');
      }

      let iterations = 0;
      let finalResponse = '';

      while (iterations < maxIterations) {
        iterations++;

        // Build messages from memory
        const messages = memory.toMessages().map(m => ({
          role: m.role as 'system' | 'user' | 'assistant',
          content: m.content,
        }));

        // Build the request
        const request: InferenceRequest = {
          model: config.model,
          messages,
          temperature: config.temperature ?? 0.7,
          stream: true,
        };

        // Stream the response
        let accumulated = '';
        const abortController = new AbortController();

        await provider.stream(request, abortController.signal, (chunk: string) => {
          accumulated += chunk;
          setStreamText(accumulated);
        });

        // Check for tool calls in the response
        const toolCall = parseToolCall(accumulated);
        if (toolCall && registry.get(toolCall.name)) {
          // Execute the tool
          memory.add('assistant', accumulated);
          try {
            const toolResult = await registry.execute(toolCall.name, toolCall.args);
            const resultStr = typeof toolResult === 'string' 
              ? toolResult 
              : JSON.stringify(toolResult);
            memory.add('user', `[Tool Result: ${toolCall.name}]: ${resultStr}`);
          } catch (err: any) {
            memory.add('user', `[Tool Error: ${toolCall.name}]: ${err.message}`);
          }
          // Continue the loop for another inference
          continue;
        }

        // No tool call — this is the final response
        finalResponse = accumulated;
        break;
      }

      // Store the final response
      memory.add('assistant', finalResponse);
      setInteractionCount(prev => prev + 1);

      return finalResponse;
    } finally {
      setLoading(false);
    }
  }

  function parseToolCall(text: string): { name: string; args: any } | null {
    // Pattern: [TOOL_CALL: tool_name({"arg": "value"})]
    const match = text.match(/\[TOOL_CALL:\s*(\w+)\((\{.*?\})\)\]/s);
    if (match) {
      try {
        return { name: match[1], args: JSON.parse(match[2]) };
      } catch {
        return null;
      }
    }
    return null;
  }

  function reset(): void {
    memory.clear();
    if (config.system) {
      memory.add('system', config.system);
    }
    batch(() => {
      setStreamText('');
      setInteractionCount(0);
    });
  }

  function dispose(): void {
    // Cleanup
  }

  return {
    memory,
    stream: streamText as ReadonlySignal<string>,
    loading: loading as ReadonlySignal<boolean>,
    interactions: interactionCount as ReadonlySignal<number>,
    send,
    reset,
    dispose,
  };
}
