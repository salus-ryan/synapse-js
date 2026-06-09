import {
  createEffect,
  createAgent,
  createTool,
  setDefaultProvider,
  createOpenAIProvider,
} from '../src/index';

const provider = createOpenAIProvider();
setDefaultProvider(provider);

async function main() {
  console.log('--- Synapse.js Agent Example ---\n');

  // 1. Define a reactive tool
  const calculator = createTool({
    name: 'calculate',
    description: 'Evaluates a mathematical expression',
    parameters: {
      type: 'object',
      properties: {
        expression: { type: 'string', description: 'The math expression (e.g. 2 + 2)' },
      },
      required: ['expression'],
    },
    execute: async ({ expression }: { expression: string }) => {
      console.log(`\n[Tool Executing: calculate(${expression})]`);
      return Function(`"use strict"; return (${expression})`)();
    },
  });

  // 2. Create the Agent
  const agent = createAgent({
    name: 'MathBot',
    model: 'gpt-4o',
    system: 'You are a math assistant. You MUST use the calculate tool for ANY math operations. Output your tool calls exactly like this: [TOOL_CALL: calculate({"expression": "2 + 2"})]. Do not try to do math yourself.',
    tools: [calculator],
  });

  // 3. Watch the agent's stream reactively
  createEffect(() => {
    const text = agent.stream();
    if (text) {
      process.stdout.write('\r\x1b[K' + text.replace(/\n/g, ' '));
    }
  });

  console.log('User: What is 12345 multiplied by 67890?');
  
  // 4. Send a message (this will trigger the loop, tool use, and final response)
  const response = await agent.send('What is 12345 multiplied by 67890?');
  
  console.log('\n\nFinal Response:', response);
}

if (process.env.OPENAI_API_KEY) {
  main().catch(console.error);
} else {
  console.log('Please set OPENAI_API_KEY to run this example.');
}
