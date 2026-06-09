import {
  createSignal,
  createEffect,
  createSynapse,
  setDefaultProvider,
  createOpenAIProvider,
} from '../src/index';

// Initialize the provider
// Requires OPENAI_API_KEY environment variable
const provider = createOpenAIProvider();
setDefaultProvider(provider);

async function main() {
  console.log('--- Synapse.js Basic Example ---\n');

  // 1. Define reactive state
  const [topic, setTopic] = createSignal('artificial intelligence');

  // 2. Define an AI Synapse (Reactive LLM Call)
  const generator = createSynapse({
    model: 'gpt-4o-mini',
    signature: 'topic -> explanation, fun_fact',
    dependencies: () => ({ topic: topic() }),
    stream: false,
  });

  // 3. Define a reactive effect to watch the output
  createEffect(() => {
    const result = generator.output();
    if (result) {
      console.log(`\nTopic: ${topic()}`);
      console.log(`Explanation: ${(result as any).explanation}`);
      console.log(`Fun Fact: ${(result as any).fun_fact}`);
    }
  });

  console.log('Triggering generation for: artificial intelligence');
  await generator.trigger();

  console.log('\nChanging topic to: quantum computing (should auto-trigger)');
  // 4. Updating the signal automatically triggers the synapse
  setTopic('quantum computing');

  // Wait for the async reaction to complete
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  generator.dispose();
}

// Only run if API key is set
if (process.env.OPENAI_API_KEY) {
  main().catch(console.error);
} else {
  console.log('Please set OPENAI_API_KEY to run this example.');
}
