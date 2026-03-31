export type HotbarFavorites = Record<string, string>;

const STORAGE_KEY = "agentic-nodes-hotbar-favorites";

function persistHotbarFavorites(favorites: HotbarFavorites): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(favorites));
}

export function getHotbarFavorites(): HotbarFavorites {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    return Object.fromEntries(
      Object.entries(parsed).filter(
        (entry): entry is [string, string] => typeof entry[0] === "string" && typeof entry[1] === "string" && entry[1].length > 0,
      ),
    );
  } catch {
    return {};
  }
}

export function setHotbarFavorite(category: string, providerId: string): HotbarFavorites {
  const next = {
    ...getHotbarFavorites(),
    [category]: providerId,
  };
  persistHotbarFavorites(next);
  return next;
}

export function clearHotbarFavorite(category: string): HotbarFavorites {
  const next = { ...getHotbarFavorites() };
  delete next[category];
  persistHotbarFavorites(next);
  return next;
}
