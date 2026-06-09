/**
 * Braille-Native AI Demo Server
 * 
 * A minimal web server that streams LLM output through the Braille codec
 * to a web page with aria-live regions for Braille display compatibility.
 * 
 * Usage: npx tsx demo-app/server.ts
 * Then open http://localhost:3333
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { readFileSync } from 'fs';
import { join } from 'path';
import { BrailleStreamEncoder, brailleTokenizer } from '../src/index';

const PORT = Number(process.env.PORT) || 3333;
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://localhost:11434/v1';
const MODEL = process.env.MODEL || 'qwen2.5:0.5b';

// ─────────────────────────────────────────────────────────────────
// SSE Streaming Endpoint
// ─────────────────────────────────────────────────────────────────

async function handleStream(req: IncomingMessage, res: ServerResponse) {
  // Parse prompt from query string
  const url = new URL(req.url!, `http://localhost:${PORT}`);
  const prompt = url.searchParams.get('prompt') || 'Hello, how are you?';
  const maxTokens = Number(url.searchParams.get('max_tokens')) || 100;

  // SSE headers
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
  });

  const encoder = new BrailleStreamEncoder({ modelId: MODEL, maxTokens });

  try {
    const response = await fetch(`${OLLAMA_URL}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ollama' },
      body: JSON.stringify({
        model: MODEL,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7,
        max_tokens: maxTokens,
        stream: true,
      }),
    });

    if (!response.ok) {
      res.write(`data: ${JSON.stringify({ type: 'error', message: `Model error: ${response.status}` })}\n\n`);
      res.end();
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) { res.end(); return; }

    const decoder = new TextDecoder();
    let tokenIndex = 0;
    const startTime = Date.now();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value, { stream: true });
      for (const line of text.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const data = line.slice(6).trim();
        if (data === '[DONE]') continue;

        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta?.content;
          if (!delta) continue;

          // Estimate confidence from timing (inter-token delay)
          const elapsed = Date.now() - startTime;
          const avgTimePerToken = elapsed / (tokenIndex + 1);
          // Faster = more confident (crude but real proxy)
          const confidence = Math.min(1.0, Math.max(0.2, 1.0 - (avgTimePerToken - 20) / 200));

          const frame = encoder.encodeToken(delta, confidence);
          tokenIndex++;

          // Send SSE event with both plain text and Braille encoding
          const event = {
            type: 'token',
            text: delta,
            braille: frame.content,
            confidence: frame.confidence,
            confidenceValue: confidence,
            position: frame.position,
            index: tokenIndex,
          };

          res.write(`data: ${JSON.stringify(event)}\n\n`);
        } catch {}
      }
    }

    reader.releaseLock();

    // Send final summary
    const stream = encoder.finalize();
    const summary = {
      type: 'done',
      totalTokens: stream.meta.totalTokens,
      avgConfidence: stream.meta.avgConfidence,
      contentBraille: stream.contentBraille,
      confidenceBraille: stream.confidenceBraille,
      duration: Date.now() - startTime,
    };
    res.write(`data: ${JSON.stringify(summary)}\n\n`);

  } catch (err: any) {
    res.write(`data: ${JSON.stringify({ type: 'error', message: err.message })}\n\n`);
  }

  res.end();
}

// ─────────────────────────────────────────────────────────────────
// Static File Server
// ─────────────────────────────────────────────────────────────────

function serveStatic(res: ServerResponse, filePath: string, contentType: string) {
  try {
    const content = readFileSync(filePath, 'utf-8');
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(content);
  } catch {
    res.writeHead(404);
    res.end('Not found');
  }
}

// ─────────────────────────────────────────────────────────────────
// Router
// ─────────────────────────────────────────────────────────────────

const server = createServer((req, res) => {
  const url = new URL(req.url!, `http://localhost:${PORT}`);

  if (url.pathname === '/api/stream') {
    handleStream(req, res);
  } else if (url.pathname === '/' || url.pathname === '/index.html') {
    serveStatic(res, join(__dirname, 'public/index.html'), 'text/html');
  } else if (url.pathname === '/style.css') {
    serveStatic(res, join(__dirname, 'public/style.css'), 'text/css');
  } else {
    res.writeHead(404);
    res.end('Not found');
  }
});

server.listen(PORT, () => {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║  Braille-Native AI Demo                                         ║
║  http://localhost:${PORT}                                          ║
║                                                                  ║
║  Model: ${MODEL.padEnd(20)}                                    ║
║  Backend: ${OLLAMA_URL.padEnd(40)}    ║
║                                                                  ║
║  This page is screen-reader compatible.                          ║
║  Braille displays will render output natively.                   ║
╚══════════════════════════════════════════════════════════════════╝
  `);
});
