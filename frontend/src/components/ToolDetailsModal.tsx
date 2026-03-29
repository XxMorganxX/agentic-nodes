import { useEffect } from "react";
import type { ChangeEvent, MouseEvent } from "react";

import {
  resolveToolNodeDetails,
  TOOL_AGENT_DESCRIPTION_CONFIG_KEY,
  TOOL_SCHEMA_TEXT_CONFIG_KEY,
  TOOL_TEMPLATE_CONFIG_KEY,
  TOOL_USER_DESCRIPTION_CONFIG_KEY,
} from "../lib/toolNodeDetails";
import { getGraphEnvVars } from "../lib/graphEnv";
import type { EditorCatalog, GraphDefinition, GraphNode } from "../lib/types";

type ToolDetailsModalProps = {
  graph: GraphDefinition;
  node: GraphNode;
  catalog: EditorCatalog | null;
  onGraphChange: (graph: GraphDefinition) => void;
  onClose: () => void;
};

function updateToolNode(
  graph: GraphDefinition,
  nodeId: string,
  updater: (node: GraphNode) => GraphNode,
): GraphDefinition {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === nodeId ? updater(node) : node)),
  };
}

export function ToolDetailsModal({
  graph,
  node,
  catalog,
  onGraphChange,
  onClose,
}: ToolDetailsModalProps) {
  const details = resolveToolNodeDetails(node, catalog, graph);
  const envVarEntries = Object.entries(getGraphEnvVars(graph));

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  function updateToolConfig(key: string, value: string) {
    onGraphChange(
      updateToolNode(graph, node.id, (currentNode) => ({
        ...currentNode,
        config: {
          ...currentNode.config,
          [key]: value,
        },
      })),
    );
  }

  function handleOverlayClick(event: MouseEvent<HTMLDivElement>) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function handleTextAreaChange(key: string) {
    return (event: ChangeEvent<HTMLTextAreaElement>) => {
      updateToolConfig(key, event.target.value);
    };
  }

  return (
    <div className="tool-details-modal-backdrop" onClick={handleOverlayClick} role="presentation">
      <section
        className="tool-details-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="tool-details-modal-title"
      >
        <div className="tool-details-modal-header">
          <div>
            <div className="tool-details-modal-eyebrow">Tool Runtime Details</div>
            <h3 id="tool-details-modal-title">
              {node.label}
              {details.toolName ? ` · ${details.toolName}` : ""}
            </h3>
            <p>
              Edit the model-facing description, schema JSON, and template used to shape the tool details exposed to
              model nodes at runtime.
            </p>
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="tool-details-modal-body">
          <label>
            User Description
            <textarea
              rows={6}
              value={details.userDescriptionText}
              onChange={handleTextAreaChange(TOOL_USER_DESCRIPTION_CONFIG_KEY)}
            />
          </label>

          <label>
            Agent Description
            <textarea
              rows={6}
              value={details.agentDescriptionText}
              onChange={handleTextAreaChange(TOOL_AGENT_DESCRIPTION_CONFIG_KEY)}
            />
          </label>

          <label>
            Schema JSON Passed To The Model
            <textarea
              rows={12}
              className="tool-details-modal-code"
              value={details.schemaText}
              onChange={handleTextAreaChange(TOOL_SCHEMA_TEXT_CONFIG_KEY)}
              spellCheck={false}
            />
          </label>

          <label>
            Template Used For The Model-Facing Tool Block
            <textarea
              rows={8}
              className="tool-details-modal-code"
              value={details.templateText}
              onChange={handleTextAreaChange(TOOL_TEMPLATE_CONFIG_KEY)}
              spellCheck={false}
            />
          </label>

          <div className="tool-details-modal-help">
            Variables available in the template: <code>{"{tool_name}"}</code>, <code>{"{tool_user_description}"}</code>
            , <code>{"{tool_agent_description}"}</code>, <code>{"{tool_description}"}</code>,{" "}
            <code>{"{tool_schema}"}</code>
          </div>

          <div className="tool-details-modal-help">
            Graph env refs can be used in any text field here:
            <div className="graph-env-reference-list">
              {envVarEntries.map(([key, value]) => (
                <code key={key} title={value}>
                  {`{${key}}`}
                </code>
              ))}
            </div>
          </div>

          {details.schemaError ? <p className="error-text">Schema JSON error: {details.schemaError}</p> : null}

          <section className="tool-details-modal-preview">
            <div className="tool-details-modal-preview-header">
              <strong>Rendered Model-Facing Preview</strong>
              <span>
                This is the resolved block built from the template, description, and schema variables for this tool
                node.
              </span>
            </div>
            <pre>{details.renderedPromptText}</pre>
          </section>
        </div>
      </section>
    </div>
  );
}
