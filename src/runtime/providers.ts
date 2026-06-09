/**
 * Model Provider Interface
 * 
 * Synapse.js is provider-agnostic. This module defines the interface
 * that any model provider must implement, plus a built-in OpenAI
 * compatible provider.
 */

// --- Types ---

export interface Message {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface InferenceRequest {
  model: string;
  messages: Message[];
  temperature?: number;
  maxTokens?: number;
  stream?: boolean;
  tools?: any[];
  responseFormat?: any;
}

export interface InferenceResponse {
  content: string;
  finishReason: string;
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface ModelProvider {
  name: string;
  /** Complete a request (non-streaming) */
  complete(request: InferenceRequest, signal?: AbortSignal): Promise<InferenceResponse>;
  /** Stream a request, calling onChunk for each token */
  stream(
    request: InferenceRequest,
    signal: AbortSignal,
    onChunk: (chunk: string) => void
  ): Promise<InferenceResponse>;
}

// --- OpenAI-Compatible Provider ---

export interface OpenAIProviderConfig {
  apiKey?: string;
  baseURL?: string;
  organization?: string;
  defaultModel?: string;
}

/**
 * Creates an OpenAI-compatible model provider.
 * Works with OpenAI, Azure OpenAI, Ollama, vLLM, and any
 * OpenAI-compatible API.
 * 
 * @example
 * const provider = createOpenAIProvider({
 *   apiKey: process.env.OPENAI_API_KEY,
 * });
 * setDefaultProvider(provider);
 */
export function createOpenAIProvider(config: OpenAIProviderConfig = {}): ModelProvider {
  const apiKey = config.apiKey || process.env.OPENAI_API_KEY || '';
  const baseURL = config.baseURL || process.env.OPENAI_API_BASE || 'https://api.openai.com/v1';

  async function complete(
    request: InferenceRequest,
    signal?: AbortSignal
  ): Promise<InferenceResponse> {
    const response = await fetch(`${baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        ...(config.organization ? { 'OpenAI-Organization': config.organization } : {}),
      },
      body: JSON.stringify({
        model: request.model,
        messages: request.messages,
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        stream: false,
      }),
      signal,
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`Provider error (${response.status}): ${errorBody}`);
    }

    const data: any = await response.json();
    const choice = data.choices[0];

    return {
      content: choice.message.content || '',
      finishReason: choice.finish_reason || 'stop',
      usage: data.usage ? {
        promptTokens: data.usage.prompt_tokens,
        completionTokens: data.usage.completion_tokens,
        totalTokens: data.usage.total_tokens,
      } : undefined,
    };
  }

  async function streamRequest(
    request: InferenceRequest,
    signal: AbortSignal,
    onChunk: (chunk: string) => void
  ): Promise<InferenceResponse> {
    const response = await fetch(`${baseURL}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
        ...(config.organization ? { 'OpenAI-Organization': config.organization } : {}),
      },
      body: JSON.stringify({
        model: request.model,
        messages: request.messages,
        temperature: request.temperature,
        max_tokens: request.maxTokens,
        stream: true,
      }),
      signal,
    });

    if (!response.ok) {
      const errorBody = await response.text();
      throw new Error(`Provider error (${response.status}): ${errorBody}`);
    }

    const reader = response.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let accumulated = '';
    let finishReason = 'stop';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = decoder.decode(value, { stream: true });
        const lines = text.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim();
            if (data === '[DONE]') continue;

            try {
              const parsed = JSON.parse(data);
              const delta = parsed.choices?.[0]?.delta?.content;
              if (delta) {
                accumulated += delta;
                onChunk(delta);
              }
              if (parsed.choices?.[0]?.finish_reason) {
                finishReason = parsed.choices[0].finish_reason;
              }
            } catch {
              // Skip malformed JSON chunks
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }

    return {
      content: accumulated,
      finishReason,
    };
  }

  return {
    name: 'openai',
    complete,
    stream: streamRequest,
  };
}

// --- Mock Provider (for testing) ---

export interface MockProviderConfig {
  /** Static response to return */
  response?: string;
  /** Response generator function */
  generator?: (request: InferenceRequest) => string;
  /** Simulated delay in ms */
  delay?: number;
  /** Simulated streaming chunk size */
  chunkSize?: number;
}

/**
 * Creates a mock provider for testing and development.
 * 
 * @example
 * const mock = createMockProvider({
 *   response: '{"sentiment": "positive", "confidence": 0.95}',
 *   delay: 100,
 * });
 */
export function createMockProvider(config: MockProviderConfig = {}): ModelProvider {
  const delay = config.delay ?? 50;
  const chunkSize = config.chunkSize ?? 5;

  function getResponse(request: InferenceRequest): string {
    if (config.generator) return config.generator(request);
    return config.response ?? '{"result": "mock response"}';
  }

  async function complete(
    request: InferenceRequest,
    signal?: AbortSignal
  ): Promise<InferenceResponse> {
    await new Promise(resolve => setTimeout(resolve, delay));
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');

    const content = getResponse(request);
    return {
      content,
      finishReason: 'stop',
      usage: {
        promptTokens: 10,
        completionTokens: Math.ceil(content.length / 4),
        totalTokens: 10 + Math.ceil(content.length / 4),
      },
    };
  }

  async function streamRequest(
    request: InferenceRequest,
    signal: AbortSignal,
    onChunk: (chunk: string) => void
  ): Promise<InferenceResponse> {
    const content = getResponse(request);
    
    // Simulate streaming by chunking the response
    for (let i = 0; i < content.length; i += chunkSize) {
      if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
      await new Promise(resolve => setTimeout(resolve, delay / 10));
      onChunk(content.slice(i, i + chunkSize));
    }

    return {
      content,
      finishReason: 'stop',
      usage: {
        promptTokens: 10,
        completionTokens: Math.ceil(content.length / 4),
        totalTokens: 10 + Math.ceil(content.length / 4),
      },
    };
  }

  return {
    name: 'mock',
    complete,
    stream: streamRequest,
  };
}
