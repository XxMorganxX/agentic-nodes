import { useEffect, useMemo, useState } from "react";
import type { ChangeEvent, MouseEvent } from "react";

import { fetchProviderDiagnostics, preflightProvider } from "../lib/api";
import { findProviderDefinition, modelProviderDefinitions, providerDefaultConfig, providerModelName } from "../lib/editor";
import { getGraphEnvVars, resolveGraphEnvReferences } from "../lib/graphEnv";
import type { EditorCatalog, GraphDefinition, GraphNode, ProviderDiagnosticsResult, ProviderPreflightResult } from "../lib/types";

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

function resolveProviderDefinition(node: GraphNode, catalog: EditorCatalog | null) {
  const providerName = String(node.config.provider_name ?? node.model_provider_name ?? "").trim();
  if (!providerName) {
    return null;
  }
  return findProviderDefinition(catalog, providerName);
}

export function ProviderDetailsModal({
  graph,
  node,
  catalog,
  onGraphChange,
  onClose,
}: ProviderDetailsModalProps) {
  const provider = resolveProviderDefinition(node, catalog);
  const availableProviders = modelProviderDefinitions(catalog);
  const envVarEntries = Object.entries(getGraphEnvVars(graph));
  const providerName = String(node.config.provider_name ?? node.model_provider_name ?? "not-set");
  const providerConfigFields = provider?.config_fields ?? [];
  const supportsLiveVerification = providerName !== "mock";
  const [preflightResult, setPreflightResult] = useState<ProviderPreflightResult | null>(null);
  const [diagnostics, setDiagnostics] = useState<ProviderDiagnosticsResult | null>(null);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [isPreflighting, setIsPreflighting] = useState(false);

  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        onClose();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onClose]);

  const preflightConfig = useMemo<Record<string, unknown>>(() => {
    const entries: Array<[string, unknown]> = [["provider_name", providerName]];
    providerConfigFields.forEach((field) => {
      entries.push([field.key, node.config[field.key]]);
    });
    return Object.fromEntries(entries);
  }, [node.config, providerConfigFields, providerName]);

  useEffect(() => {
    let cancelled = false;
    if (!providerName || providerName === "not-set") {
      setPreflightResult(null);
      setDiagnostics(null);
      setPreflightError(null);
      return () => {
        cancelled = true;
      };
    }

    setIsPreflighting(true);
    setPreflightError(null);
    Promise.all([
      preflightProvider(providerName, preflightConfig, false),
      fetchProviderDiagnostics(providerName, preflightConfig, false),
    ])
      .then(([result, diagnosticsResult]) => {
        if (!cancelled) {
          setPreflightResult(result);
          setDiagnostics(diagnosticsResult);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setPreflightError("Unable to load provider health.");
          setPreflightResult(null);
          setDiagnostics(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsPreflighting(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [preflightConfig, providerName]);

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
    return (event: ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
      updateProviderConfig(key, event.target.value);
    };
  }

  function handleNumberInputChange(key: string) {
    return (event: ChangeEvent<HTMLInputElement>) => {
      updateProviderConfig(key, event.target.value === "" ? "" : Number(event.target.value));
    };
  }

  function handleProviderChange(nextProviderName: string) {
    const nextProvider = findProviderDefinition(catalog, nextProviderName);
    if (!nextProvider) {
      return;
    }
    const nextProviderConfig = providerDefaultConfig(nextProvider);
    const providerConfigKeys = Array.from(
      new Set(
        availableProviders.flatMap((candidate) => [
          "provider_name",
          ...((candidate.config_fields ?? []).map((field) => field.key)),
        ]),
      ),
    );
    onGraphChange(
      updateModelNode(graph, node.id, (currentNode) => {
        const nextConfig = { ...currentNode.config };
        providerConfigKeys.forEach((key) => delete nextConfig[key]);
        return {
          ...currentNode,
          model_provider_name: nextProviderName,
          config: {
            ...nextConfig,
            ...nextProviderConfig,
            provider_name: nextProviderName,
          },
        };
      }),
    );
  }

  async function handleLiveVerification() {
    setIsPreflighting(true);
    setPreflightError(null);
    try {
      const [result, diagnosticsResult] = await Promise.all([
        preflightProvider(providerName, preflightConfig, true),
        fetchProviderDiagnostics(providerName, preflightConfig, true),
      ]);
      setPreflightResult(result);
      setDiagnostics(diagnosticsResult);
    } catch {
      setPreflightError("Live provider verification failed.");
      setPreflightResult(null);
      setDiagnostics(null);
    } finally {
      setIsPreflighting(false);
    }
  }

  const resolvedPreviewConfig = Object.fromEntries(
    [["provider_name", providerName], ...providerConfigFields.map((field) => [field.key, node.config[field.key]])].map(
      ([key, value]) => [
        key,
        typeof value === "string" ? resolveGraphEnvReferences(value, graph) || null : (value ?? null),
      ],
    ),
  );

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
            {preflightResult ? (
              <div className="tool-details-modal-help">
                <strong>Provider Health</strong>
                <div>{preflightResult.message}</div>
                {preflightResult.warnings?.map((warning) => (
                  <div key={warning}>{warning}</div>
                ))}
              </div>
            ) : null}
            {diagnostics ? (
              <div className="tool-details-modal-help">
                <strong>Provider Diagnostics</strong>
                <div className="provider-diagnostics-card">
                  <div className="provider-diagnostics-section">
                    <div className="provider-diagnostics-section-title">Backend</div>
                    <div className="provider-diagnostics-row">
                      <span>Active backend</span>
                      <strong>{diagnostics.active_backend}</strong>
                    </div>
                    <div className="provider-diagnostics-row">
                      <span>Authentication status</span>
                      <strong>{diagnostics.authentication_status}</strong>
                    </div>
                  </div>
                  {diagnostics.active_backend === "claude_code" ? (
                    <div className="provider-diagnostics-section">
                      <div className="provider-diagnostics-section-title">Claude Code</div>
                      <div className="provider-diagnostics-row">
                        <span>Claude binary</span>
                        <strong>{diagnostics.claude_binary_exists ? "found" : "not found"}</strong>
                      </div>
                    </div>
                  ) : null}
                  {diagnostics.active_backend === "claude_code" || diagnostics.active_backend === "anthropic_api" ? (
                    <div className="provider-diagnostics-section">
                      <div className="provider-diagnostics-section-title">Environment</div>
                      <div className="provider-diagnostics-row">
                        <span>`ANTHROPIC_API_KEY` present</span>
                        <strong>{diagnostics.anthropic_api_key_present ? "yes" : "no"}</strong>
                      </div>
                    </div>
                  ) : null}
                {diagnostics.child_env_sanitized ? (
                    <div className="provider-diagnostics-section">
                      <div className="provider-diagnostics-section-title">Child Process</div>
                      <div className="provider-diagnostics-list">
                        <div>Sanitized environment enabled.</div>
                        <div>Strips: {diagnostics.sanitized_env_removed_vars.join(", ")}</div>
                      </div>
                    </div>
                ) : null}
                  {diagnostics.warning ? (
                    <div className="provider-diagnostics-section">
                      <div className="provider-diagnostics-section-title">Warning</div>
                      <div className="provider-diagnostics-list">
                        <div>{diagnostics.warning}</div>
                      </div>
                    </div>
                  ) : null}
                </div>
              </div>
            ) : null}
            {preflightError ? <div className="tool-details-modal-help">{preflightError}</div> : null}
            {!supportsLiveVerification ? (
              <div className="tool-details-modal-help">Live verification is not required for the mock provider.</div>
            ) : null}
            <button
              type="button"
              className="secondary-button"
              onClick={handleLiveVerification}
              disabled={isPreflighting || !supportsLiveVerification}
            >
              {isPreflighting ? "Checking Provider..." : supportsLiveVerification ? "Run Live Verification" : "Live Verification Not Required"}
            </button>
          </section>

          <div className="provider-details-grid">
            <label>
              Provider
              <select
                value={providerName}
                onChange={(event) => handleProviderChange(event.target.value)}
              >
                {availableProviders.map((candidate) => {
                  const candidateName = providerModelName(candidate);
                  return (
                    <option key={candidate.provider_id} value={candidateName}>
                      {candidate.display_name}
                    </option>
                  );
                })}
              </select>
            </label>
            {providerConfigFields.map((field) => (
              <label key={field.key}>
                {field.label}
                {(() => {
                  const currentValue = String(node.config[field.key] ?? "");
                  const isSelectField = field.input_type === "select" && (field.options?.length ?? 0) > 0;
                  const isModelSelectField = isSelectField && field.key === "model";
                  const selectOptions =
                    isSelectField && currentValue && !field.options?.some((option) => option.value === currentValue)
                      ? [...(field.options ?? []), { value: currentValue, label: `Custom: ${currentValue}` }]
                      : (field.options ?? []);
                  const datalistId = `${node.id}-${field.key}-modal-options`;
                  return (
                    <>
                      {isModelSelectField ? (
                        <>
                          <input
                            list={datalistId}
                            value={currentValue}
                            placeholder={field.placeholder || "Select or type a model id"}
                            onChange={handleTextInputChange(field.key)}
                          />
                          <datalist id={datalistId}>
                            {selectOptions.map((option) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </datalist>
                        </>
                      ) : isSelectField ? (
                        <select
                          value={currentValue}
                          onChange={handleTextInputChange(field.key)}
                        >
                          {selectOptions.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <input
                          type={field.input_type === "number" ? "number" : "text"}
                          value={currentValue}
                          placeholder={field.placeholder || undefined}
                          onChange={
                            field.input_type === "number"
                              ? handleNumberInputChange(field.key)
                              : handleTextInputChange(field.key)
                          }
                        />
                      )}
                    </>
                  );
                })()}
              </label>
            ))}
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
