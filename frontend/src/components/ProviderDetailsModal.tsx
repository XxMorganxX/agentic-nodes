import { useEffect } from "react";
import type { ChangeEvent, MouseEvent } from "react";

import { getGraphEnvVars, resolveGraphEnvReferences } from "../lib/graphEnv";
import type { EditorCatalog, GraphDefinition, GraphNode, NodeProviderDefinition } from "../lib/types";

type ProviderDetailsModalProps = {
  graph: GraphDefinition;
  node: GraphNode;
  catalog: EditorCatalog | null;
  onGraphChange: (graph: GraphDefinition) => void;
  onClose: () => void;
};

function updateModelNode(
  graph: GraphDefinition,
  nodeId: string,
  updater: (node: GraphNode) => GraphNode,
): GraphDefinition {
  return {
    ...graph,
    nodes: graph.nodes.map((node) => (node.id === nodeId ? updater(node) : node)),
  };
}

function resolveProviderDefinition(node: GraphNode, catalog: EditorCatalog | null): NodeProviderDefinition | null {
  const providerName = String(node.config.provider_name ?? node.model_provider_name ?? "").trim();
  if (!providerName) {
    return null;
  }
  return (
    (catalog?.node_providers ?? []).find(
      (provider) => provider.category === "provider" && provider.provider_id === `provider.${providerName}`,
    ) ?? null
  );
}

export function ProviderDetailsModal({
  graph,
  node,
  catalog,
  onGraphChange,
  onClose,
}: ProviderDetailsModalProps) {
  const provider = resolveProviderDefinition(node, catalog);
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

  function updateProviderConfig(key: string, value: string | number) {
    onGraphChange(
      updateModelNode(graph, node.id, (currentNode) => ({
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

  function handleTextInputChange(key: string) {
    return (event: ChangeEvent<HTMLInputElement>) => {
      updateProviderConfig(key, event.target.value);
    };
  }

  function handleNumberInputChange(key: string) {
    return (event: ChangeEvent<HTMLInputElement>) => {
      updateProviderConfig(key, event.target.value === "" ? "" : Number(event.target.value));
    };
  }

  const providerName = String(node.config.provider_name ?? node.model_provider_name ?? "not-set");
  const providerModel = String(node.config.model ?? "");
  const resolvedPreviewConfig = {
    provider_name: resolveGraphEnvReferences(String(node.config.provider_name ?? node.model_provider_name ?? ""), graph) || null,
    model: resolveGraphEnvReferences(String(node.config.model ?? ""), graph) || null,
    temperature: node.config.temperature ?? null,
    max_tokens: node.config.max_tokens ?? null,
    api_base: resolveGraphEnvReferences(String(node.config.api_base ?? ""), graph) || null,
    api_key_env_var: resolveGraphEnvReferences(String(node.config.api_key_env_var ?? ""), graph) || null,
  };

  return (
    <div className="tool-details-modal-backdrop" onClick={handleOverlayClick} role="presentation">
      <section
        className="tool-details-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="provider-details-modal-title"
      >
        <div className="tool-details-modal-header">
          <div>
            <div className="tool-details-modal-eyebrow">API Provider Details</div>
            <h3 id="provider-details-modal-title">
              {node.label}
              {provider ? ` · ${provider.display_name}` : ""}
            </h3>
            <p>
              Required provider selection stays on the API node. Use this modal to review provider capabilities and tune
              optional provider parameters for the selected API step.
            </p>
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>
            Close
          </button>
        </div>

        <div className="tool-details-modal-body">
          <section className="provider-details-summary">
            <div className="provider-details-summary-header">
              <strong>Selected Provider</strong>
              <span>{provider?.display_name ?? providerName}</span>
            </div>
            <p>{provider?.description ?? "No provider description is available for the current selection."}</p>
            {provider?.capabilities.length ? (
              <div className="provider-details-capabilities">
                {provider.capabilities.map((capability) => (
                  <span key={capability} className="provider-capability-chip">
                    {capability}
                  </span>
                ))}
              </div>
            ) : null}
          </section>

          <div className="provider-details-grid">
            <label>
              Provider Name
              <input value={providerName} readOnly />
            </label>

            <label>
              Model
              <input value={providerModel} onChange={handleTextInputChange("model")} />
            </label>

            <label>
              Temperature
              <input
                type="number"
                step="0.1"
                value={String(node.config.temperature ?? "")}
                onChange={handleNumberInputChange("temperature")}
              />
            </label>

            <label>
              Max Tokens
              <input
                type="number"
                value={String(node.config.max_tokens ?? "")}
                onChange={handleNumberInputChange("max_tokens")}
              />
            </label>

            <label>
              API Base
              <input value={String(node.config.api_base ?? "")} onChange={handleTextInputChange("api_base")} />
            </label>

            <label>
              API Key Env Var
              <input
                value={String(node.config.api_key_env_var ?? "")}
                onChange={handleTextInputChange("api_key_env_var")}
              />
            </label>
          </div>

          <div className="tool-details-modal-help">
            Required provider choice is controlled from the API node itself. These fields are optional overrides for the
            selected provider.
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

          <section className="tool-details-modal-preview">
            <div className="tool-details-modal-preview-header">
              <strong>Resolved Provider Config</strong>
              <span>This preview shows provider settings after graph env references are substituted.</span>
            </div>
            <pre>{JSON.stringify(resolvedPreviewConfig, null, 2)}</pre>
          </section>
        </div>
      </section>
    </div>
  );
}
