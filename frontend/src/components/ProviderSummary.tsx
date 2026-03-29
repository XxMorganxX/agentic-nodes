import { useState } from "react";

import type { EditorCatalog, NodeProviderDefinition } from "../lib/types";

type ProviderSummaryProps = {
  catalog: EditorCatalog | null;
  variant?: "panel" | "drawer";
  query?: string;
  onQueryChange?: (query: string) => void;
  onProviderClick?: (provider: NodeProviderDefinition) => void;
};

const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  start: "Start nodes are entry points that inject outside signals, requests, or webhook payloads into a graph run.",
  api: "API nodes call external or provider-backed services such as LLM endpoints while keeping a stable graph role.",
  tool: "Tool nodes perform structured actions with validated inputs and routable success or failure results.",
  data: "Data nodes gather, transform, or enrich deterministic context inside the graph.",
  end: "End nodes terminate the run and shape the final output returned by the system.",
};

const CATEGORY_ORDER = ["all", "start", "api", "tool", "data", "end"] as const;

const QUICK_PICK_PROVIDER_IDS = ["core.input", "core.api", "tool.registry", "core.output"];

const KIND_LABELS: Record<string, string> = {
  input: "IN",
  model: "API",
  provider: "PROV",
  tool: "TOOL",
  data: "DATA",
  output: "END",
};

const KIND_GLYPHS: Record<string, string> = {
  input: "IN",
  model: "AI",
  provider: "PR",
  tool: "FX",
  data: "DB",
  output: "OUT",
};

function compactDescription(description: string): string {
  const sentence = description.split(".")[0]?.trim() ?? "";
  return sentence.length > 0 ? sentence : description;
}

function groupProviders(providers: NodeProviderDefinition[]): Array<[string, NodeProviderDefinition[]]> {
  const grouped = new Map<string, NodeProviderDefinition[]>();

  providers.forEach((provider) => {
    const existing = grouped.get(provider.category) ?? [];
    existing.push(provider);
    grouped.set(provider.category, existing);
  });

  return Array.from(grouped.entries());
}

export function ProviderSummary({
  catalog,
  variant = "panel",
  query = "",
  onQueryChange,
  onProviderClick,
}: ProviderSummaryProps) {
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const normalizedQuery = query.trim().toLowerCase();
  const allProviders = (catalog?.node_providers ?? []).filter((provider) => provider.category !== "provider");
  const providers = allProviders.filter((provider) => {
    if (!normalizedQuery) {
      return activeCategory === "all" ? true : provider.category === activeCategory;
    }
    const haystack = [
      provider.display_name,
      provider.provider_id,
      provider.node_kind,
      provider.description,
      provider.capabilities.join(" "),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(normalizedQuery) && (activeCategory === "all" ? true : provider.category === activeCategory);
  });
  const quickPicks = QUICK_PICK_PROVIDER_IDS
    .map((providerId) => allProviders.find((provider) => provider.provider_id === providerId))
    .filter((provider): provider is NodeProviderDefinition => Boolean(provider))
    .filter((provider) => (activeCategory === "all" ? true : provider.category === activeCategory))
    .filter((provider) => {
      if (!normalizedQuery) {
        return true;
      }
      const haystack = [
        provider.display_name,
        provider.provider_id,
        provider.node_kind,
        provider.description,
        provider.capabilities.join(" "),
      ]
        .join(" ")
        .toLowerCase();
      return haystack.includes(normalizedQuery);
    });
  const groupedProviders = groupProviders(providers);

  return (
    <section className={variant === "drawer" ? "provider-summary provider-summary--drawer" : "panel provider-summary"}>
      {variant === "panel" ? (
        <div className="panel-header">
          <h2>Node Categories</h2>
          <p>Drag a node card onto the canvas to create a new node in the agent graph.</p>
        </div>
      ) : null}
      <div className={`provider-groups${variant === "drawer" ? " provider-groups--drawer" : ""}`}>
        {onQueryChange ? (
          <div className="provider-library-shell">
            <section className="provider-library-hero">
              <div>
                <div className="provider-library-eyebrow">Node Library</div>
                <h4>Build the graph with quick visual picks</h4>
                <p>Click a tile to insert it near the center, or drag it directly onto the canvas.</p>
              </div>
            </section>
            <label className="provider-search provider-search--library">
              <span>Search</span>
              <input
                value={query}
                onChange={(event) => onQueryChange(event.target.value)}
                placeholder="Search nodes, ids, or capabilities"
              />
            </label>
            <div className="provider-category-chips" role="tablist" aria-label="Node categories">
              {CATEGORY_ORDER.map((category) => (
                <button
                  key={category}
                  type="button"
                  className={`provider-category-chip${activeCategory === category ? " is-active" : ""}`}
                  onClick={() => setActiveCategory(category)}
                >
                  {category}
                </button>
              ))}
            </div>
            {quickPicks.length > 0 ? (
              <section className="provider-quick-picks">
                <div className="provider-section-heading">
                  <strong>Quick Add</strong>
                  <span>Common starting points</span>
                </div>
                <div className="provider-quick-pick-list">
                  {quickPicks.map((provider) => (
                    <button
                      key={provider.provider_id}
                      type="button"
                      className="provider-quick-pick"
                      onClick={() => onProviderClick?.(provider)}
                    >
                      <span className={`provider-kind-pill provider-kind-pill--${provider.node_kind}`}>
                        {KIND_LABELS[provider.node_kind] ?? provider.node_kind.slice(0, 3).toUpperCase()}
                      </span>
                      <span className="provider-quick-pick-copy">
                        <strong>{provider.display_name}</strong>
                        <span>{provider.provider_id}</span>
                      </span>
                    </button>
                  ))}
                </div>
              </section>
            ) : null}
          </div>
        ) : null}
        {providers.length === 0 ? (
          <div className="empty-panel">
            {normalizedQuery ? "No nodes match the current search." : "No node definitions available."}
          </div>
        ) : (
          groupedProviders.map(([category, categoryProviders]) => (
            <article key={category} className={`provider-group${variant === "drawer" ? " provider-group--drawer" : ""}`}>
              <div className="provider-section-heading">
                <strong>{category}</strong>
                <span>{CATEGORY_DESCRIPTIONS[category] ?? "Custom node category."}</span>
              </div>
              {catalog?.contracts[category] ? (
                <div className="provider-contract">
                  <span>Accepts: {catalog.contracts[category].accepted_inputs.join(", ")}</span>
                  <span>Produces: {catalog.contracts[category].produced_outputs.join(", ")}</span>
                </div>
              ) : null}
              <div className={`provider-list${variant === "drawer" ? " provider-list--tiles" : ""}`}>
                {categoryProviders.map((provider) => (
                  <section
                    key={provider.provider_id}
                    className={`provider-item provider-item-draggable${variant === "drawer" ? " provider-item--drawer" : ""}`}
                    draggable
                    onDragStart={(event) => {
                      event.dataTransfer.setData("application/graph-node-provider", JSON.stringify(provider));
                      event.dataTransfer.effectAllowed = "copy";
                    }}
                    onClick={() => onProviderClick?.(provider)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        onProviderClick?.(provider);
                      }
                    }}
                    tabIndex={onProviderClick ? 0 : -1}
                    role={onProviderClick ? "button" : undefined}
                  >
                    {variant === "drawer" ? (
                      <>
                        <div className="provider-drawer-tile-top">
                          <div className={`provider-visual-mark provider-visual-mark--${provider.node_kind}`}>
                            <span>{KIND_GLYPHS[provider.node_kind] ?? provider.node_kind.slice(0, 2).toUpperCase()}</span>
                          </div>
                          <span className={`provider-kind-pill provider-kind-pill--${provider.node_kind}`}>
                            {KIND_LABELS[provider.node_kind] ?? provider.node_kind.slice(0, 3).toUpperCase()}
                          </span>
                        </div>
                        <div className="provider-item-header">
                          <strong>{provider.display_name}</strong>
                        </div>
                        <p>{compactDescription(provider.description)}.</p>
                        {onProviderClick ? <div className="provider-action-hint">Add or drag</div> : null}
                      </>
                    ) : (
                      <>
                        <div className="provider-tile-top">
                          <span className={`provider-kind-pill provider-kind-pill--${provider.node_kind}`}>
                            {KIND_LABELS[provider.node_kind] ?? provider.node_kind.slice(0, 3).toUpperCase()}
                          </span>
                          <code>{provider.provider_id}</code>
                        </div>
                        <div className="provider-item-header">
                          <strong>{provider.display_name}</strong>
                        </div>
                        <p>{provider.description}</p>
                        <div className="provider-meta">
                          <span>Kind: {provider.node_kind}</span>
                        </div>
                        {provider.capabilities.length > 0 ? (
                          <div className="provider-capability-list">
                            {provider.capabilities.slice(0, 3).map((capability) => (
                              <span key={capability} className="provider-capability-chip">
                                {capability}
                              </span>
                            ))}
                          </div>
                        ) : null}
                        {onProviderClick ? <div className="provider-action-hint">Click to add or drag</div> : null}
                      </>
                    )}
                  </section>
                ))}
              </div>
            </article>
          ))
        )}
      </div>
      {catalog?.tools.length && variant !== "drawer" ? (
        <div className="provider-groups">
          <article className="provider-group">
            <h3>registered tools</h3>
            <p>Model nodes can expose these tools at prompt time and route to compatible tool nodes.</p>
            <div className="provider-list">
              {catalog.tools.map((tool) => (
                <section key={tool.name} className="provider-item">
                  <div className="provider-item-header">
                    <strong>{tool.name}</strong>
                  </div>
                  <p>{tool.description}</p>
                </section>
              ))}
            </div>
          </article>
        </div>
      ) : null}
    </section>
  );
}
