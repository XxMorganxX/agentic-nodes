import { useEffect } from "react";
import type { ChangeEvent, MouseEvent } from "react";

import { getNodeInstanceLabel } from "../lib/nodeInstanceLabels";
import { insertTokenAtEnd, listPromptBlockAvailableVariables, PROMPT_BLOCK_STARTERS, renderPromptBlockPreview } from "../lib/promptBlockEditor";
import type { GraphDefinition, GraphNode, RunState } from "../lib/types";

type PromptBlockDetailsModalProps = {
  graph: GraphDefinition;
  node: GraphNode;
  runState: RunState | null;
  onGraphChange: (graph: GraphDefinition) => void;
  onClose: () => void;
};

function updatePromptBlockNode(
  graph: GraphDefinition,
  nodeId: string,
  updater: (node: GraphNode) => GraphNode,
): GraphDefinition {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === nodeId ? updater(node) : node)),
  };
}

export function PromptBlockDetailsModal({
  graph,
  node,
  runState,
  onGraphChange,
  onClose,
}: PromptBlockDetailsModalProps) {
  const nodeLabel = getNodeInstanceLabel(graph, node);
  const availableVariables = listPromptBlockAvailableVariables(graph);
  const renderedPreview = renderPromptBlockPreview(node, graph, runState);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  function updatePromptBlockConfig(updater: (config: GraphNode["config"]) => GraphNode["config"]) {
    onGraphChange(
      updatePromptBlockNode(graph, node.id, (currentNode) => ({
        ...currentNode,
        config: updater(currentNode.config),
      })),
    );
  }

  function handleOverlayClick(event: MouseEvent<HTMLDivElement>) {
    if (event.target === event.currentTarget) {
      onClose();
    }
  }

  function handleRoleChange(event: ChangeEvent<HTMLSelectElement>) {
    updatePromptBlockConfig((config) => ({
      ...config,
      role: event.target.value,
      mode: "prompt_block",
    }));
  }

  function handleNameChange(event: ChangeEvent<HTMLInputElement>) {
    updatePromptBlockConfig((config) => ({
      ...config,
      name: event.target.value,
      mode: "prompt_block",
    }));
  }

  function handleContentChange(event: ChangeEvent<HTMLTextAreaElement>) {
    updatePromptBlockConfig((config) => ({
      ...config,
      content: event.target.value,
      mode: "prompt_block",
    }));
  }

  function handleInsertStarter() {
    updatePromptBlockConfig((config) => {
      const role = String(config.role ?? "user");
      const fallback = PROMPT_BLOCK_STARTERS[role] ?? PROMPT_BLOCK_STARTERS.user;
      const existingContent = String(config.content ?? "");
      return {
        ...config,
        content: existingContent.trim().length > 0 ? existingContent : fallback,
        mode: "prompt_block",
      };
    });
  }

  function handleInsertVariable(token: string) {
    updatePromptBlockConfig((config) => ({
      ...config,
      content: insertTokenAtEnd(String(config.content ?? ""), `{${token}}`),
      mode: "prompt_block",
    }));
  }

  return (
    <div className="tool-details-modal-backdrop" onClick={handleOverlayClick} role="presentation">
      <section
        className="tool-details-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="prompt-block-details-modal-title"
      >
        <div className="tool-details-modal-header">
          <div>
            <div className="tool-details-modal-eyebrow">Prompt Block Details</div>
            <h3 id="prompt-block-details-modal-title">{nodeLabel}</h3>
            <p>Edit this message block in a dedicated modal and preview the rendered prompt with current graph variables.</p>
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="tool-details-modal-body">
          <div className="context-builder-placeholder-bar">
            <button type="button" className="secondary-button context-builder-inline-button" onClick={handleInsertStarter}>
              Insert Starter
            </button>
            {availableVariables.map((token) => (
              <button
                key={token}
                type="button"
                className="secondary-button context-builder-token-button"
                onClick={() => handleInsertVariable(token)}
              >
                {`{${token}}`}
              </button>
            ))}
          </div>

          <label>
            Message Role
            <select value={String(node.config.role ?? "user")} onChange={handleRoleChange}>
              <option value="system">system</option>
              <option value="user">user</option>
              <option value="assistant">assistant</option>
            </select>
          </label>

          <label>
            Message Name
            <input
              value={String(node.config.name ?? "")}
              placeholder="Optional label for the message block"
              onChange={handleNameChange}
            />
          </label>

          <label>
            Message Content
            <textarea
              rows={8}
              value={String(node.config.content ?? "")}
              placeholder="Enter the message content to inject into downstream prompt assembly."
              onChange={handleContentChange}
            />
          </label>

          <div className="tool-details-modal-help">
            Available variables: {availableVariables.length > 0 ? availableVariables.join(", ") : "None"}
          </div>

          <section className="tool-details-modal-preview">
            <div className="tool-details-modal-preview-header">
              <strong>Rendered Preview</strong>
              <span>This shows how the prompt block resolves after variable substitution.</span>
            </div>
            <pre>{renderedPreview}</pre>
          </section>
        </div>
      </section>
    </div>
  );
}
