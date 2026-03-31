import { useMemo, useState } from "react";

import { getGraphEnvVars, STANDARD_GRAPH_ENV_FIELDS } from "../lib/graphEnv";
import type { GraphDocument } from "../lib/types";

type GraphEnvEditorProps = {
  graph: GraphDocument | null;
  onGraphChange: (graph: GraphDocument) => void;
};

const GRAPH_ENV_KEY_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const STANDARD_GRAPH_ENV_KEYS: ReadonlySet<string> = new Set(STANDARD_GRAPH_ENV_FIELDS.map((field) => field.key));
const SENSITIVE_ENV_KEY_PATTERN = /(password|passwd|passphrase|secret|token|api[_-]?key|private[_-]?key|credential)/i;

function isSensitiveEnvKey(key: string): boolean {
  return SENSITIVE_ENV_KEY_PATTERN.test(key);
}

function isEnvValueVisible(revealedEnvKeys: Record<string, boolean>, key: string): boolean {
  return revealedEnvKeys[key] ?? !isSensitiveEnvKey(key);
}

function updateGraphEnvVars(
  graph: GraphDocument,
  updater: (envVars: Record<string, string>) => Record<string, string>,
): GraphDocument {
  return {
    ...graph,
    env_vars: updater(getGraphEnvVars(graph)),
  };
}

export function GraphEnvEditor({ graph, onGraphChange }: GraphEnvEditorProps) {
  const [newEnvKey, setNewEnvKey] = useState("");
  const [newEnvValue, setNewEnvValue] = useState("");
  const [revealedEnvKeys, setRevealedEnvKeys] = useState<Record<string, boolean>>({});
  const [newEnvValueVisible, setNewEnvValueVisible] = useState(false);

  const envVars = useMemo(() => getGraphEnvVars(graph), [graph]);
  const customEnvEntries = useMemo(
    () => Object.entries(envVars).filter(([key]) => !STANDARD_GRAPH_ENV_KEYS.has(key)),
    [envVars],
  );
  const trimmedNewEnvKey = newEnvKey.trim();
  const newEnvValueInputVisible = newEnvValueVisible || !isSensitiveEnvKey(trimmedNewEnvKey);
  const newEnvKeyError =
    trimmedNewEnvKey.length === 0
      ? null
      : GRAPH_ENV_KEY_PATTERN.test(trimmedNewEnvKey)
        ? null
        : "Use letters, numbers, and underscores only.";

  if (!graph) {
    return null;
  }

  return (
    <div className="env-tiles">
      {STANDARD_GRAPH_ENV_FIELDS.map((field) => (
        <div key={field.key} className="env-tile">
          <label className="env-tile-label">{field.label}</label>
          <div className="env-tile-value-row">
            <input
              type={isEnvValueVisible(revealedEnvKeys, field.key) ? "text" : "password"}
              value={envVars[field.key] ?? ""}
              placeholder={field.placeholder}
              onChange={(event) =>
                onGraphChange(
                  updateGraphEnvVars(graph, (currentEnvVars) => ({
                    ...currentEnvVars,
                    [field.key]: event.target.value,
                  })),
                )
              }
            />
            <button
              type="button"
              className="secondary-button env-tile-visibility-toggle"
              onClick={() =>
                setRevealedEnvKeys((currentValue) => ({
                  ...currentValue,
                  [field.key]: !isEnvValueVisible(currentValue, field.key),
                }))
              }
              aria-label={`${isEnvValueVisible(revealedEnvKeys, field.key) ? "Hide" : "Show"} value for ${field.key}`}
              aria-pressed={isEnvValueVisible(revealedEnvKeys, field.key)}
            >
              {isEnvValueVisible(revealedEnvKeys, field.key) ? "Hide" : "Show"}
            </button>
          </div>
        </div>
      ))}
      {customEnvEntries.map(([key, value]) => (
        <div key={key} className="env-tile">
          <label className="env-tile-label">
            <code>{`{${key}}`}</code>
          </label>
          <div className="env-tile-value-row">
            <input
              type={isEnvValueVisible(revealedEnvKeys, key) ? "text" : "password"}
              value={value}
              onChange={(event) =>
                onGraphChange(
                  updateGraphEnvVars(graph, (currentEnvVars) => ({
                    ...currentEnvVars,
                    [key]: event.target.value,
                  })),
                )
              }
            />
            <button
              type="button"
              className="secondary-button env-tile-visibility-toggle"
              onClick={() =>
                setRevealedEnvKeys((currentValue) => ({
                  ...currentValue,
                  [key]: !isEnvValueVisible(currentValue, key),
                }))
              }
              aria-label={`${isEnvValueVisible(revealedEnvKeys, key) ? "Hide" : "Show"} value for ${key}`}
              aria-pressed={isEnvValueVisible(revealedEnvKeys, key)}
            >
              {isEnvValueVisible(revealedEnvKeys, key) ? "Hide" : "Show"}
            </button>
            <button
              type="button"
              className="secondary-button env-tile-remove"
              onClick={() =>
                onGraphChange(
                  updateGraphEnvVars(graph, (currentEnvVars) => {
                    const nextEnvVars = { ...currentEnvVars };
                    delete nextEnvVars[key];
                    return nextEnvVars;
                  }),
                )
              }
            >
              &times;
            </button>
          </div>
        </div>
      ))}
      <div className="env-tile env-tile--add">
        <div className="env-tile-add-inputs">
          <input
            value={newEnvKey}
            placeholder="VAR_NAME"
            onChange={(event) => setNewEnvKey(event.target.value)}
          />
          <input
            type={newEnvValueInputVisible ? "text" : "password"}
            value={newEnvValue}
            placeholder="value"
            onChange={(event) => setNewEnvValue(event.target.value)}
          />
          <button
            type="button"
            className="secondary-button env-tile-visibility-toggle"
            onClick={() => setNewEnvValueVisible((currentValue) => !currentValue)}
            aria-label={`${newEnvValueInputVisible ? "Hide" : "Show"} new environment variable value`}
            aria-pressed={newEnvValueInputVisible}
          >
            {newEnvValueInputVisible ? "Hide" : "Show"}
          </button>
        </div>
        <button
          type="button"
          className="secondary-button"
          onClick={() => {
            if (!trimmedNewEnvKey || newEnvKeyError) {
              return;
            }
            onGraphChange(
              updateGraphEnvVars(graph, (currentEnvVars) => ({
                ...currentEnvVars,
                [trimmedNewEnvKey]: newEnvValue,
              })),
            );
            setNewEnvKey("");
            setNewEnvValue("");
            setNewEnvValueVisible(false);
          }}
          disabled={!trimmedNewEnvKey || Boolean(newEnvKeyError)}
        >
          + Add
        </button>
        {newEnvKeyError ? <p className="env-tile-error">{newEnvKeyError}</p> : null}
      </div>
    </div>
  );
}
