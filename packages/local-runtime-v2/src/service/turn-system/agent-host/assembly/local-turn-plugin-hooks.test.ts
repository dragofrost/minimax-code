import { afterEach, describe, expect, it, vi } from 'vitest';
import type { PiEventWriter } from '@mavis/agent-core/pi-turn-runner';
import { mergePluginHookDecisions } from '@mavis/plugin-hooks';
import {
  createLocalPluginHookEventReporter,
  createLocalPluginPostLlmHook,
  createLocalPluginPreLlmHook,
  emitLocalPluginHookWarnings,
  localPluginHookCoordinator,
} from './local-turn-plugin-hooks.js';

describe('Plugin Hook display messages', () => {
  it('deduplicates diagnostics but retains system messages across tool calls', async () => {
    const appendEvents = vi.fn(
      async (_events: Parameters<PiEventWriter['appendEvents']>[0]) => undefined,
    );
    const reporter = createLocalPluginHookEventReporter({
      writer: { appendEvents, pushRuntime: vi.fn() },
      sessionId: 'session-dedupe',
      turnId: 'turn-dedupe',
    });
    const result = {
      decision: {
        decision: 'allow' as const,
        systemMessage: 'initial system message',
        terminalSequence: '\u0007',
      },
      diagnostics: [
        {
          code: 'HOOK_INVALID_INPUT' as const,
          pluginName: 'compatible-tools',
          event: 'PostToolUse' as const,
          sourcePath: '/plugin/hook.mjs',
          declarationOrder: 0,
        },
      ],
    };
    const repeatedResult = {
      ...result,
      decision: {
        ...result.decision,
        systemMessage: 'a later system message from the same event category',
        terminalSequence: '\u001b[0m',
      },
    };

    await emitLocalPluginHookWarnings({
      reporter,
      sessionId: 'session-dedupe',
      turnId: 'turn-dedupe',
      event: 'PostToolUse',
      result,
    });
    await emitLocalPluginHookWarnings({
      reporter,
      sessionId: 'session-dedupe',
      turnId: 'turn-dedupe',
      event: 'PostToolUse',
      result: repeatedResult,
    });

    await emitLocalPluginHookWarnings({
      reporter,
      sessionId: 'session-dedupe',
      turnId: 'turn-dedupe',
      event: 'PostToolUse',
      result: repeatedResult,
    });

    expect(appendEvents).toHaveBeenCalledTimes(3);
    const emitted = appendEvents.mock.calls.flatMap(([events]) =>
      events.map((event) => {
        const envelope = JSON.parse(event.payload.stream_resp as string);
        return {
          id: envelope.agent_message.msg_id,
          ...JSON.parse(envelope.agent_message.msg_content),
        };
      }),
    );
    const messages = emitted.filter((event) => event.category === 'system-message');
    expect(messages.map((event) => event.message)).toEqual([
      'initial system message',
      'a later system message from the same event category',
      'a later system message from the same event category',
    ]);
    expect(new Set(messages.map((event) => event.id)).size).toBe(3);
    expect(emitted.filter((event) => event.category === 'diagnostic')).toHaveLength(1);
    expect(emitted.filter((event) => event.category === 'terminal-control')).toHaveLength(1);
    const serialized = JSON.stringify(appendEvents.mock.calls[0]?.[0]);
    expect(serialized).toContain('was skipped');
    expect(serialized).toContain('tool execution and result were not changed');
    expect(serialized).toContain('HOOK_INVALID_INPUT');
  });

  it('preserves identical text in different categories and keeps delivery failure best-effort', async () => {
    const appendEvents = vi.fn<PiEventWriter['appendEvents']>().mockResolvedValue(undefined);
    const reporter = createLocalPluginHookEventReporter({
      writer: { appendEvents, pushRuntime: vi.fn() },
      sessionId: 'category-session',
      turnId: 'category-turn',
    });
    const input = {
      reporter,
      sessionId: 'category-session',
      turnId: 'category-turn',
      event: 'Stop' as const,
      message: 'same text',
      decision: { decision: 'allow' as const, systemMessage: 'same text' },
    };
    await emitLocalPluginHookWarnings(input);
    expect(appendEvents.mock.calls[0]?.[0]).toHaveLength(2);
    appendEvents.mockRejectedValueOnce(new Error('writer unavailable'));
    await expect(emitLocalPluginHookWarnings(input)).resolves.toBeUndefined();
    expect(input.decision).toEqual({
      decision: 'allow',
      systemMessage: 'same text',
    });
  });
});

describe('Plugin LLM lifecycle hooks', () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('keeps the current user turn even when PreLLM filters older history', async () => {
    vi.spyOn(localPluginHookCoordinator, 'runEvent').mockResolvedValue({
      decision: { decision: 'allow', keepMessageIndexes: [1] },
      diagnostics: [],
    });
    const signal = new AbortController().signal;
    const executionInput = {
      pluginHooks: [{ event: 'PreLLM' }],
      lease: { sessionId: 'llm-session', turnId: 'llm-turn', signal },
      session: { workspaceDir: '/workspace' },
      pluginHookRuntimeContext: { model: 'test-model' },
    } as never;
    const hook = createLocalPluginPreLlmHook(executionInput);
    expect(hook).toBeDefined();

    const messages = [
      { role: 'user', content: [{ type: 'text', text: 'old question' }], timestamp: 1 },
      { role: 'assistant', content: [{ type: 'text', text: 'old answer' }], timestamp: 2 },
      { role: 'user', content: [{ type: 'text', text: 'current question' }], timestamp: 3 },
      { role: 'assistant', content: [{ type: 'text', text: 'current work' }], timestamp: 4 },
    ];
    const decision = await hook?.({
      sessionId: 'llm-session',
      turnId: 'llm-turn',
      phase: 'iteration',
      messages: messages as never,
      canonicalMessages: messages as never,
      model: {} as never,
      systemPrompt: 'system',
      tools: [],
      thinkingLevel: 'off',
      signal,
    });

    expect(decision).toEqual({
      type: 'replaceRequestMessages',
      messages: [messages[1], messages[2], messages[3]],
      reason: 'plugin-pre-llm-context-filter',
    });
  });

  it('maps a stopping PostLLM Hook to a fail decision before tools execute', async () => {
    vi.spyOn(localPluginHookCoordinator, 'runEvent').mockResolvedValue({
      decision: {
        decision: 'allow',
        continue: false,
        stopReason: 'response rejected by policy',
      },
      diagnostics: [],
    });
    const signal = new AbortController().signal;
    const hook = createLocalPluginPostLlmHook({
      pluginHooks: [{ event: 'PostLLM' }],
      lease: { sessionId: 'post-session', turnId: 'post-turn', signal },
      session: { workspaceDir: '/workspace' },
      pluginHookRuntimeContext: { model: 'test-model' },
    } as never);
    expect(hook).toBeDefined();

    const decision = await hook?.({
      sessionId: 'post-session',
      turnId: 'post-turn',
      message: {} as never,
      messages: [],
      signal,
    });

    expect(decision).toEqual({
      type: 'fail',
      reason: 'response rejected by policy',
    });
  });

  it('unions retained indexes from multiple PreLLM Hook decisions', () => {
    expect(
      mergePluginHookDecisions(
        { decision: 'allow', keepMessageIndexes: [1, 4] },
        { decision: 'allow', keepMessageIndexes: [2, 4] },
      ).keepMessageIndexes,
    ).toEqual([1, 2, 4]);
  });
});
