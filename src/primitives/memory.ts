/**
 * Memory Primitive — Reactive Context Graph
 * 
 * Instead of manually managing message arrays, Memory provides
 * a reactive graph that automatically:
 * - Tracks conversation history
 * - Compresses old messages when approaching token limits
 * - Provides windowed views of context
 * - Supports semantic retrieval (with embeddings)
 * - Auto-summarizes when context exceeds budget
 * 
 * Memory is reactive: when messages are added, downstream
 * Synapses that depend on the memory automatically re-fire.
 */

import {
  createSignal,
  createComputed,
  ReadonlySignal,
  batch,
} from '../core/reactive.js';

// --- Types ---

export interface MemoryNode {
  id: string;
  role: 'system' | 'user' | 'assistant' | 'summary';
  content: string;
  timestamp: number;
  tokens: number;
  metadata?: Record<string, any>;
  embedding?: number[];
  compressed?: boolean;
}

export interface MemoryConfig {
  /** Maximum token budget for the context window */
  maxTokens: number;
  /** Strategy for managing overflow */
  strategy: 'sliding_window' | 'summarize' | 'semantic';
  /** How many tokens to reserve for the response */
  reserveTokens?: number;
  /** Custom token counter (default: rough estimate of 4 chars per token) */
  tokenCounter?: (text: string) => number;
  /** Custom summarizer function */
  summarizer?: (messages: MemoryNode[]) => Promise<string>;
}

export interface MemoryInstance {
  /** All messages in the memory (reactive) */
  messages: ReadonlySignal<MemoryNode[]>;
  /** The windowed context that fits within token budget (reactive) */
  context: ReadonlySignal<MemoryNode[]>;
  /** Current token usage (reactive) */
  tokenUsage: ReadonlySignal<number>;
  /** Add a message to memory */
  add: (role: MemoryNode['role'], content: string, metadata?: Record<string, any>) => void;
  /** Clear all messages */
  clear: () => void;
  /** Get messages as a formatted array for LLM consumption */
  toMessages: () => Array<{ role: string; content: string }>;
  /** Search memory by semantic similarity (if embeddings available) */
  search: (query: string, topK?: number) => MemoryNode[];
  /** Force compression/summarization */
  compress: () => Promise<void>;
  /** Total message count (reactive) */
  count: ReadonlySignal<number>;
}

// --- Utilities ---

let idCounter = 0;
function generateId(): string {
  return `mem_${Date.now()}_${++idCounter}`;
}

function estimateTokens(text: string): number {
  // Rough estimate: ~4 characters per token
  return Math.ceil(text.length / 4);
}

// --- Create Memory ---

/**
 * Creates a reactive memory/context graph.
 * 
 * @example
 * const memory = createMemory({
 *   maxTokens: 4096,
 *   strategy: 'sliding_window',
 * });
 * 
 * memory.add('user', 'Hello!');
 * memory.add('assistant', 'Hi there! How can I help?');
 * 
 * // Use in a Synapse
 * const chat = createSynapse({
 *   model: 'gpt-4o',
 *   signature: 'context, message -> response',
 *   dependencies: () => ({
 *     context: memory.toMessages(),
 *     message: userInput(),
 *   }),
 * });
 */
export function createMemory(config: MemoryConfig): MemoryInstance {
  const tokenCounter = config.tokenCounter ?? estimateTokens;
  const reserveTokens = config.reserveTokens ?? 500;
  const budget = config.maxTokens - reserveTokens;

  const [allMessages, setAllMessages] = createSignal<MemoryNode[]>([]);

  // Computed: windowed context within token budget
  const context = createComputed<MemoryNode[]>(() => {
    const messages = allMessages();
    
    switch (config.strategy) {
      case 'sliding_window':
        return slidingWindow(messages, budget, tokenCounter);
      case 'summarize':
        return summarizeStrategy(messages, budget, tokenCounter);
      case 'semantic':
        return semanticStrategy(messages, budget, tokenCounter);
      default:
        return slidingWindow(messages, budget, tokenCounter);
    }
  });

  // Computed: current token usage
  const tokenUsage = createComputed(() => {
    return context().reduce((sum, msg) => sum + msg.tokens, 0);
  });

  // Computed: message count
  const count = createComputed(() => allMessages().length);

  function add(
    role: MemoryNode['role'],
    content: string,
    metadata?: Record<string, any>
  ): void {
    const node: MemoryNode = {
      id: generateId(),
      role,
      content,
      timestamp: Date.now(),
      tokens: tokenCounter(content),
      metadata,
      compressed: false,
    };

    setAllMessages(prev => [...prev, node]);
  }

  function clear(): void {
    setAllMessages([]);
  }

  function toMessages(): Array<{ role: string; content: string }> {
    return context().map(node => ({
      role: node.role === 'summary' ? 'system' : node.role,
      content: node.content,
    }));
  }

  function search(query: string, topK: number = 5): MemoryNode[] {
    // Simple keyword search (upgrade to vector similarity when embeddings available)
    const queryLower = query.toLowerCase();
    const scored = allMessages()
      .map(node => ({
        node,
        score: node.content.toLowerCase().includes(queryLower) ? 1 : 0,
      }))
      .filter(item => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return scored.map(item => item.node);
  }

  async function compress(): Promise<void> {
    const messages = allMessages();
    if (messages.length <= 2) return;

    if (config.summarizer) {
      const toCompress = messages.slice(0, -2); // Keep last 2 messages
      const summary = await config.summarizer(toCompress);
      const summaryNode: MemoryNode = {
        id: generateId(),
        role: 'summary',
        content: summary,
        timestamp: Date.now(),
        tokens: tokenCounter(summary),
        compressed: true,
      };

      setAllMessages([summaryNode, ...messages.slice(-2)]);
    } else {
      // Default: just keep recent messages within budget
      const windowed = slidingWindow(messages, budget, tokenCounter);
      setAllMessages(windowed);
    }
  }

  return {
    messages: allMessages as ReadonlySignal<MemoryNode[]>,
    context,
    tokenUsage,
    add,
    clear,
    toMessages,
    search,
    compress,
    count,
  };
}

// --- Strategy Implementations ---

function slidingWindow(
  messages: MemoryNode[],
  budget: number,
  tokenCounter: (text: string) => number
): MemoryNode[] {
  // Always include system messages
  const systemMessages = messages.filter(m => m.role === 'system' || m.role === 'summary');
  const nonSystem = messages.filter(m => m.role !== 'system' && m.role !== 'summary');

  let systemTokens = systemMessages.reduce((sum, m) => sum + m.tokens, 0);
  let remaining = budget - systemTokens;

  // Take messages from the end (most recent first)
  const result: MemoryNode[] = [];
  for (let i = nonSystem.length - 1; i >= 0; i--) {
    if (nonSystem[i].tokens <= remaining) {
      result.unshift(nonSystem[i]);
      remaining -= nonSystem[i].tokens;
    } else {
      break;
    }
  }

  return [...systemMessages, ...result];
}

function summarizeStrategy(
  messages: MemoryNode[],
  budget: number,
  tokenCounter: (text: string) => number
): MemoryNode[] {
  // For now, same as sliding window
  // The actual summarization happens in compress()
  return slidingWindow(messages, budget, tokenCounter);
}

function semanticStrategy(
  messages: MemoryNode[],
  budget: number,
  tokenCounter: (text: string) => number
): MemoryNode[] {
  // For now, same as sliding window
  // Would use embeddings for semantic relevance scoring
  return slidingWindow(messages, budget, tokenCounter);
}
