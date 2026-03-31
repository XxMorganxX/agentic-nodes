---
status: investigating
trigger: "claude-code-provider-health: error_max_turns misclassified as provider failure"
created: 2026-03-30
updated: 2026-03-30
---

## Current Focus
hypothesis: Health check treats any `is_error` or non-success result as failure; `error_max_turns` is a bounded probe outcome, not connectivity/config failure
test: Trace health endpoint + claude_code adapter success criteria
expecting: Exact branch that maps result to "failed"
next_action: Grep and read health + provider code paths

## Symptoms
expected: Provider health should accurately report whether claude_code is usable; turn-limit response should not be misclassified as failure
actual: Shows failed request with subtype error_max_turns, stop_reason tool_use, errors max turns (1)
errors: error_max_turns / Reached maximum number of turns (1)
reproduction: Trigger Provider Health for claude_code in app
started: unknown

## Eliminated

## Evidence
- `ClaudeCodeCLIModelProvider._run_command` raises on any `returncode != 0` with message prefix `claude_code provider request failed` (matches user-visible error).
- `_healthcheck_command` builds the same CLI invocation as normal runs: `--json-schema` + `--max-turns` from `provider_config` (default 1 via `_number_config` or 1).
- User payload: `subtype: error_max_turns`, `stop_reason: tool_use`, `errors: ["Reached maximum number of turns (1)"]` — model needed more than one turn to finish structured JSON (tool/stop pattern), not a missing CLI or auth failure.
- No code references `is_error`, `subtype`, or `error_max_turns` (grep); classification is binary: process exit code + exception text.

## Resolution
root_cause: Live auth healthcheck uses `--max-turns` default 1 with `--json-schema`; Claude Code can stop with `tool_use` on turn 1 and need another turn to emit JSON, triggering CLI `error_max_turns` and non-zero exit; `_run_command` maps any non-zero exit to generic failure.
fix: (not applied — readonly investigation)
verification: (pending)
files_changed: []
