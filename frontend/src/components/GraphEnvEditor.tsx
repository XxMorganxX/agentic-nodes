import { useMemo, useState } from "react";

import { getGraphEnvVars, STANDARD_GRAPH_ENV_FIELDS } from "../lib/graphEnv";
import type { GraphDefinition } from "../lib/types";

type GraphEnvEditorProps = {
  graph: GraphDefinition | null;
  onGraphChange: (graph: GraphDefinition) => void;
};

const GRAPH_ENV_KEY_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const STANDARD_GRAPH_ENV_KEYS: ReadonlySet<string> = new Set(STANDARD_GRAPH_ENV_FIELDS.map((field) => field.key));

function updateGraphEnvVars(
  graph: GraphDefinition,
  updater: (envVars: Record<string, string>) => Record<string, string>,
): GraphDefinition {
  return {
    ...graph,
    env_vars: updater(getGraphEnvVars(graph)),
  };
}

export function GraphEnvEditor({ graph, onGraphChange }: GraphEnvEditorProps) {
  const [newEnvKey, setNewEnvKey] = useState("");
  const [newEnvValue, setNewEnvValue] = useState("");

  const envVars = useMemo(() => getGraphEnvVars(graph), [graph]);
  const customEnvEntries = useMemo(
    () => Object.entries(envVars).filter(([key]) => !STANDARD_GRAPH_ENV_KEYS.has(key)),
    [envVars],
  );
  const trimmedNewEnvKey = newEnvKey.trim();
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
          <input
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
        </div>
      ))}
      {customEnvEntries.map(([key, value]) => (
        <div key={key} className="env-tile">
          <label className="env-tile-label"><code>{`{${key}}`}</code></label>
          <div className="env-tile-value-row">
            <input
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
            value={newEnvValue}
            placeholder="value"
            onChange={(event) => setNewEnvValue(event.target.value)}
          />
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
