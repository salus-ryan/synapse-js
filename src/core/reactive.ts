/**
 * Synapse.js Core Reactive Engine
 * 
 * A fine-grained reactivity system designed for AI inference.
 * Unlike traditional signals that hold synchronous values, Synapse signals
 * natively support async streams, token-by-token updates, and cancellation.
 */

// --- Types ---

export type Subscriber = () => void;
export type Cleanup = () => void;
export type Unsubscribe = () => void;

export interface SignalOptions<T> {
  equals?: (prev: T, next: T) => boolean;
  name?: string;
}

export interface ReadonlySignal<T> {
  (): T;
  peek: () => T;
  subscribe: (fn: Subscriber) => Unsubscribe;
  name?: string;
}

export interface WritableSignal<T> extends ReadonlySignal<T> {
  set: (value: T) => void;
  update: (fn: (prev: T) => T) => void;
}

// --- Internals ---

let currentObserver: Set<Set<Subscriber>> | null = null;
let batchDepth = 0;
const pendingEffects = new Set<Subscriber>();

/**
 * Batch multiple signal updates into a single flush.
 * Effects only run after the outermost batch completes.
 */
export function batch(fn: () => void): void {
  batchDepth++;
  try {
    fn();
  } finally {
    batchDepth--;
    if (batchDepth === 0) {
      flushEffects();
    }
  }
}

function flushEffects(): void {
  const effects = [...pendingEffects];
  pendingEffects.clear();
  for (const effect of effects) {
    effect();
  }
}

function notify(subscribers: Set<Subscriber>): void {
  // Copy subscribers before iterating to prevent infinite loops
  // when effects re-subscribe during notification
  const subs = [...subscribers];
  for (const sub of subs) {
    if (batchDepth > 0) {
      pendingEffects.add(sub);
    } else {
      sub();
    }
  }
}

// --- Signal ---

/**
 * Creates a reactive signal — the fundamental state primitive.
 * 
 * @example
 * const [count, setCount] = createSignal(0);
 * console.log(count()); // 0
 * setCount(1);
 * console.log(count()); // 1
 */
export function createSignal<T>(
  initialValue: T,
  options?: SignalOptions<T>
): [ReadonlySignal<T>, (value: T | ((prev: T) => T)) => void] {
  let value = initialValue;
  const subscribers = new Set<Subscriber>();
  const equals = options?.equals ?? Object.is;

  const getter = () => {
    // Track dependency
    if (currentObserver) {
      currentObserver.add(subscribers);
    }
    return value;
  };

  const read: ReadonlySignal<T> = Object.assign(
    getter,
    {
      peek: () => value,
      subscribe: (fn: Subscriber): Unsubscribe => {
        subscribers.add(fn);
        return () => subscribers.delete(fn);
      },
    }
  );

  if (options?.name) {
    Object.defineProperty(read, 'name', { value: options.name, writable: false });
  }

  const write = (next: T | ((prev: T) => T)) => {
    const nextValue = typeof next === 'function' 
      ? (next as (prev: T) => T)(value) 
      : next;
    if (!equals(value, nextValue)) {
      value = nextValue;
      notify(subscribers);
    }
  };

  return [read, write];
}

// --- Computed ---

/**
 * Creates a derived reactive value that automatically tracks dependencies.
 * Re-evaluates only when its dependencies change.
 * 
 * @example
 * const [name, setName] = createSignal("world");
 * const greeting = createComputed(() => `Hello, ${name()}!`);
 * console.log(greeting()); // "Hello, world!"
 */
export function createComputed<T>(
  fn: () => T,
  options?: SignalOptions<T>
): ReadonlySignal<T> {
  let value: T;
  let dirty = true;
  const subscribers = new Set<Subscriber>();
  let trackedDeps = new Set<Set<Subscriber>>();
  const equals = options?.equals ?? Object.is;

  const recompute = () => {
    dirty = true;
    notify(subscribers);
  };

  const getter = () => {
    if (dirty) {
      // Unsubscribe from old deps
      for (const dep of trackedDeps) {
        dep.delete(recompute);
      }
      trackedDeps = new Set();

      // Track new deps
      const prevObserver = currentObserver;
      currentObserver = trackedDeps;
      const newValue = fn();
      currentObserver = prevObserver;

      // Subscribe to new deps
      for (const dep of trackedDeps) {
        dep.add(recompute);
      }

      if (!equals(value!, newValue)) {
        value = newValue;
      }
      dirty = false;
    }

    // Track this computed as a dependency of the outer observer
    if (currentObserver) {
      currentObserver.add(subscribers);
    }

    return value;
  };

  const read: ReadonlySignal<T> = Object.assign(
    getter,
    {
      peek: () => {
        if (dirty) {
          read(); // Force evaluation
        }
        return value;
      },
      subscribe: (sub: Subscriber): Unsubscribe => {
        subscribers.add(sub);
        return () => subscribers.delete(sub);
      },
    }
  );

  if (options?.name) {
    Object.defineProperty(read, 'name', { value: options.name, writable: false });
  }

  return read;
}

// --- Effect ---

/**
 * Creates a reactive side effect that re-runs when its dependencies change.
 * Returns a cleanup function to dispose the effect.
 * 
 * @example
 * const [count, setCount] = createSignal(0);
 * const dispose = createEffect(() => {
 *   console.log("Count is:", count());
 * });
 */
export function createEffect(fn: () => void | Cleanup): Cleanup {
  let trackedDeps = new Set<Set<Subscriber>>();
  let cleanup: void | Cleanup;
  let disposed = false;
  let running = false;

  const execute = () => {
    if (disposed) return;
    if (running) return; // Prevent re-entrance
    running = true;

    try {
      // Run cleanup from previous execution
      if (cleanup) {
        cleanup();
        cleanup = undefined;
      }

      // Unsubscribe from old deps
      for (const dep of trackedDeps) {
        dep.delete(execute);
      }
      trackedDeps = new Set();

      // Track new deps
      const prevObserver = currentObserver;
      currentObserver = trackedDeps;
      cleanup = fn();
      currentObserver = prevObserver;

      // Subscribe to new deps
      for (const dep of trackedDeps) {
        dep.add(execute);
      }
    } finally {
      running = false;
    }
  };

  // Initial execution
  execute();

  // Return dispose function
  return () => {
    disposed = true;
    if (cleanup) cleanup();
    for (const dep of trackedDeps) {
      dep.delete(execute);
    }
    trackedDeps.clear();
  };
}

// --- Async Signal ---

export interface AsyncSignalState<T> {
  value: T | undefined;
  error: Error | undefined;
  loading: boolean;
  streaming: boolean;
}

/**
 * Creates an async signal that represents a value resolved over time.
 * Supports streaming (token-by-token updates) and cancellation.
 * 
 * This is the bridge between synchronous reactivity and async AI inference.
 */
export function createAsyncSignal<T>(
  initialValue?: T
): {
  state: ReadonlySignal<AsyncSignalState<T>>;
  resolve: (value: T) => void;
  reject: (error: Error) => void;
  stream: (chunk: T, merge: (prev: T | undefined, chunk: T) => T) => void;
  startLoading: () => void;
  startStreaming: () => void;
  reset: () => void;
} {
  const [state, setState] = createSignal<AsyncSignalState<T>>({
    value: initialValue,
    error: undefined,
    loading: false,
    streaming: false,
  });

  return {
    state,
    resolve: (value: T) => {
      setState({ value, error: undefined, loading: false, streaming: false });
    },
    reject: (error: Error) => {
      setState({ value: state.peek().value, error, loading: false, streaming: false });
    },
    stream: (chunk: T, merge: (prev: T | undefined, chunk: T) => T) => {
      const current = state.peek();
      setState({
        value: merge(current.value, chunk),
        error: undefined,
        loading: false,
        streaming: true,
      });
    },
    startLoading: () => {
      setState({ ...state.peek(), loading: true, streaming: false, error: undefined });
    },
    startStreaming: () => {
      setState({ ...state.peek(), loading: false, streaming: true, error: undefined });
    },
    reset: () => {
      setState({ value: initialValue, error: undefined, loading: false, streaming: false });
    },
  };
}
