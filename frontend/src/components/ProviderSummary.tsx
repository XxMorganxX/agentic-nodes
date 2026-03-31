import { useState } from "react";

import { useRenderDiagnostics } from "../lib/dragDiagnostics";
import type { SavedNode } from "../lib/savedNodes";
import type { EditorCatalog, NodeProviderDefinition } from "../lib/types";

type ProviderSummaryProps = {
  catalog: EditorCatalog | null;
  variant?: "panel" | "drawer";
  query?: string;
  onQueryChange?: (query: string) => void;
  onProviderClick?: (provider: NodeProviderDefinition) => void;
  hotbarItems?: {
    hotkey: string;
    label: string;
    category: string;
    provider: NodeProviderDefinition | null;
    isFavorite: boolean;
  }[];
  hotbarFavorites?: Record<string, string>;
  onToggleHotbarFavorite?: (provider: NodeProviderDefinition) => void;
  savedNodes?: SavedNode[];
  onSavedNodeClick?: (saved: SavedNode) => void;
  onDeleteSavedNode?: (id: string) => void;
};

type HotbarItem = NonNullable<ProviderSummaryProps["hotbarItems"]>[number];
type VisibleHotbarItem = Omit<HotbarItem, "provider"> & { provider: NodeProviderDefinition };

const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  start: "Start nodes are entry points that inject run-button input or external events such as Discord messages into a graph run.",
  api: "API nodes call external or provider-backed services such as LLM endpoints while keeping a stable graph role.",
  tool: "Tool nodes perform structured actions with validated inputs and routable success or failure results.",
  data: "Data nodes gather, transform, or enrich deterministic context inside the graph.",
  end: "End nodes terminate the run and shape the final output returned by the system.",
};

const CATEGORY_ORDER = ["all", "saved", "start", "api", "tool", "data", "end"] as const;

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
  hotbarItems = [],
  hotbarFavorites = {},
  onToggleHotbarFavorite,
  savedNodes = [],
  onSavedNodeClick,
  onDeleteSavedNode,
}: ProviderSummaryProps) {
  const [activeCategory, setActiveCategory] = useState<string>("all");
  const normalizedQuery = query.trim().toLowerCase();
  const allProviders = (catalog?.node_providers ?? []).filter(
    (provider) => provider.category !== "provider" && provider.provider_id !== "core.input",
  );
  const isSavedCategory = activeCategory === "saved";
  const providers = isSavedCategory
    ? []
    : allProviders.filter((provider) => {
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
  const filteredSavedNodes = savedNodes.filter((saved) => {
    if (isSavedCategory || activeCategory === "all") {
      if (!normalizedQuery) return true;
      const haystack = [saved.name, saved.kind, saved.category, saved.description, saved.provider_id].join(" ").toLowerCase();
      return haystack.includes(normalizedQuery);
    }
    if (saved.category !== activeCategory) return false;
    if (!normalizedQuery) return true;
    const haystack = [saved.name, saved.kind, saved.category, saved.description, saved.provider_id].join(" ").toLowerCase();
    return haystack.includes(normalizedQuery);
  });
  const visibleHotbarItems = hotbarItems.filter((item): item is VisibleHotbarItem => {
    if (!item.provider) {
      return false;
    }
    if (activeCategory !== "all" && item.category !== activeCategory) {
      return false;
    }
    if (!normalizedQuery) {
      return true;
    }
    const haystack = [
      item.label,
      item.category,
      item.provider.display_name,
      item.provider.provider_id,
      item.provider.description,
      item.provider.capabilities.join(" "),
    ]
      .join(" ")
      .toLowerCase();
    return haystack.includes(normalizedQuery);
  });
  const groupedProviders = groupProviders(providers);

  useRenderDiagnostics(
    "ProviderSummary",
    variant === "drawer",
    {
      variant,
      activeCategory,
      queryLength: query.length,
      providerCount: providers.length,
      savedCount: filteredSavedNodes.length,
      quickPickCount: visibleHotbarItems.length,
    },
    12,
  );

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
                <p>Click a tile to insert it near the center, drag it onto the canvas, or star it to pin that category hotbar slot.</p>
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
            {visibleHotbarItems.length > 0 ? (
              <section className="provider-quick-picks">
                <div className="provider-section-heading">
                  <strong>Hotbar</strong>
                  <span>Starred nodes replace their category slot here and in quick add.</span>
                </div>
                <div className="provider-quick-pick-list">
                  {visibleHotbarItems.map((item) => (
                    <button
                      key={item.hotkey}
                      type="button"
                      className="provider-quick-pick"
                      onClick={() => item.provider && onProviderClick?.(item.provider)}
                      disabled={!item.provider}
                    >
                      <span className={`provider-kind-pill provider-kind-pill--${item.provider.node_kind}`}>
                        {KIND_LABELS[item.provider.node_kind] ?? item.provider.node_kind.slice(0, 3).toUpperCase()}
                      </span>
                      <span className="provider-quick-pick-copy">
                        <strong>{item.label}</strong>
                        <span>{item.provider.display_name}</span>
                      </span>
                      {item.isFavorite ? <span className="provider-hotbar-badge">Pinned</span> : null}
                    </button>
                  ))}
                </div>
              </section>
            ) : null}
          </div>
        ) : null}
        {filteredSavedNodes.length > 0 ? (
          <article className={`provider-group${variant === "drawer" ? " provider-group--drawer" : ""}`}>
            <div className="provider-section-heading">
              <strong>Saved Nodes</strong>
              <span>Reuse your saved node configurations.</span>
            </div>
            <div className={`provider-list${variant === "drawer" ? " provider-list--tiles" : ""}`}>
              {filteredSavedNodes.map((saved) => (
                <section
                  key={saved.id}
                  className={`provider-item provider-item-draggable saved-node-item${variant === "drawer" ? " provider-item--drawer" : ""}`}
                  draggable
                  onDragStart={(event) => {
                    event.dataTransfer.setData("application/graph-saved-node", JSON.stringify(saved));
                    event.dataTransfer.effectAllowed = "copy";
                  }}
                  onClick={() => onSavedNodeClick?.(saved)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSavedNodeClick?.(saved);
                    }
                  }}
                  tabIndex={onSavedNodeClick ? 0 : -1}
                  role={onSavedNodeClick ? "button" : undefined}
                >
                  {variant === "drawer" ? (
                    <>
                      <div className="provider-drawer-tile-top">
                        <div className={`provider-visual-mark provider-visual-mark--${saved.kind}`}>
                          <span>{KIND_GLYPHS[saved.kind] ?? saved.kind.slice(0, 2).toUpperCase()}</span>
                        </div>
                        <span className={`provider-kind-pill provider-kind-pill--${saved.kind}`}>
                          {KIND_LABELS[saved.kind] ?? saved.kind.slice(0, 3).toUpperCase()}
                        </span>
                        <span className="saved-node-badge">Saved</span>
                      </div>
                      <div className="provider-item-header">
                        <strong>{saved.name}</strong>
                      </div>
                      <p>{saved.description ? compactDescription(saved.description) + "." : `Saved ${saved.kind} node.`}</p>
                      <div className="saved-node-actions">
                        <span className="provider-action-hint">Add or drag</span>
                        {onDeleteSavedNode ? (
                          <button
                            type="button"
                            className="saved-node-delete"
                            onClick={(event) => {
                              event.stopPropagation();
                              onDeleteSavedNode(saved.id);
                            }}
                            aria-label={`Delete saved node ${saved.name}`}
                            title="Remove from library"
                          >
                            <svg viewBox="0 0 24 24" aria-hidden="true">
                              <path d="M6 7h12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M10 11v5M14 11v5M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-12" />
                            </svg>
                          </button>
                        ) : null}
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="provider-tile-top">
                        <span className={`provider-kind-pill provider-kind-pill--${saved.kind}`}>
                          {KIND_LABELS[saved.kind] ?? saved.kind.slice(0, 3).toUpperCase()}
                        </span>
                        <span className="saved-node-badge">Saved</span>
                      </div>
                      <div className="provider-item-header">
                        <strong>{saved.name}</strong>
                      </div>
                      <p>{saved.description || `Saved ${saved.kind} node configuration.`}</p>
                      <div className="saved-node-actions">
                        {onSavedNodeClick ? <span className="provider-action-hint">Click to add or drag</span> : null}
                        {onDeleteSavedNode ? (
                          <button
                            type="button"
                            className="saved-node-delete"
                            onClick={(event) => {
                              event.stopPropagation();
                              onDeleteSavedNode(saved.id);
                            }}
                            aria-label={`Delete saved node ${saved.name}`}
                            title="Remove from library"
                          >
                            <svg viewBox="0 0 24 24" aria-hidden="true">
                              <path d="M6 7h12M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M10 11v5M14 11v5M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2l1-12" />
                            </svg>
                          </button>
                        ) : null}
                      </div>
                    </>
                  )}
                </section>
              ))}
            </div>
          </article>
        ) : isSavedCategory ? (
          <div className="empty-panel">
            {normalizedQuery ? "No saved nodes match the current search." : "No saved nodes yet. Save a node from the inspector to reuse it here."}
          </div>
        ) : null}
        {isSavedCategory ? null : providers.length === 0 && filteredSavedNodes.length === 0 ? (
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
                {categoryProviders.map((provider) => {
                  const isHotbarFavorite = hotbarFavorites[provider.category] === provider.provider_id;
                  return (
                    <section
                      key={provider.provider_id}
                      className={`provider-item provider-item-draggable${variant === "drawer" ? " provider-item--drawer" : ""}${isHotbarFavorite ? " is-hotbar-favorite" : ""}`}
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
                            {onToggleHotbarFavorite ? (
                              <button
                                type="button"
                                className={`provider-favorite-toggle${isHotbarFavorite ? " is-active" : ""}`}
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onToggleHotbarFavorite(provider);
                                }}
                                onPointerDown={(event) => event.stopPropagation()}
                                aria-pressed={isHotbarFavorite}
                                aria-label={
                                  isHotbarFavorite
                                    ? `Remove ${provider.display_name} from ${provider.category} hotbar`
                                    : `Add ${provider.display_name} to ${provider.category} hotbar`
                                }
                                title={isHotbarFavorite ? "Remove from hotbar" : "Add to hotbar"}
                              >
                                <svg viewBox="0 0 24 24" aria-hidden="true">
                                  <path d="M12 3.5l2.6 5.27 5.82.85-4.21 4.1.99 5.8L12 16.75 6.8 19.52l.99-5.8-4.21-4.1 5.82-.85L12 3.5z" />
                                </svg>
                              </button>
                            ) : null}
                          </div>
                          <div className="provider-item-header">
                            <strong>{provider.display_name}</strong>
                          </div>
                          <p>{compactDescription(provider.description)}.</p>
                          {onProviderClick ? (
                            <div className="provider-action-hint">
                              {isHotbarFavorite ? "Pinned to hotbar. Add or drag" : "Add or drag"}
                            </div>
                          ) : null}
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
                  );
                })}
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
