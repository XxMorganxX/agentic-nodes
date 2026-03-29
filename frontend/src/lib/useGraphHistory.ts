import { useCallback, useEffect, useRef, useState } from "react";

import type { GraphDefinition } from "./types";

const MAX_HISTORY = 80;
const NO_PENDING_QUIET_UPDATE = Symbol("NO_PENDING_QUIET_UPDATE");

export type GraphHistory = {
  graph: GraphDefinition | null;
  /** Update graph and push previous state onto the undo stack. */
  set: (next: GraphDefinition | null) => void;
  /** Update graph without recording history (use for intermediate drags). */
  setQuiet: (next: GraphDefinition | null) => void;
  undo: () => void;
  redo: () => void;
  canUndo: boolean;
  canRedo: boolean;
  reset: (next: GraphDefinition | null) => void;
};

export function useGraphHistory(initial: GraphDefinition | null = null): GraphHistory {
  const [graph, setGraph] = useState<GraphDefinition | null>(initial);
  const undoStack = useRef<(GraphDefinition | null)[]>([]);
  const redoStack = useRef<(GraphDefinition | null)[]>([]);
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const quietBaselineRef = useRef<GraphDefinition | null | typeof NO_PENDING_QUIET_UPDATE>(NO_PENDING_QUIET_UPDATE);

  const syncFlags = useCallback(() => {
    setCanUndo(undoStack.current.length > 0);
    setCanRedo(redoStack.current.length > 0);
  }, []);

  const set = useCallback(
    (next: GraphDefinition | null) => {
      setGraph((prev) => {
        const baseline = quietBaselineRef.current === NO_PENDING_QUIET_UPDATE ? prev : quietBaselineRef.current;
        quietBaselineRef.current = NO_PENDING_QUIET_UPDATE;

        if (baseline !== next) {
          undoStack.current.push(baseline);
          if (undoStack.current.length > MAX_HISTORY) {
            undoStack.current.splice(0, undoStack.current.length - MAX_HISTORY);
          }
          redoStack.current = [];
        }
        return next;
      });
      syncFlags();
    },
    [syncFlags],
  );

  const undo = useCallback(() => {
    const prev = undoStack.current.pop();
    if (prev === undefined) return;
    quietBaselineRef.current = NO_PENDING_QUIET_UPDATE;
    setGraph((current) => {
      redoStack.current.push(current);
      syncFlags();
      return prev;
    });
  }, [syncFlags]);

  const redo = useCallback(() => {
    const next = redoStack.current.pop();
    if (next === undefined) return;
    quietBaselineRef.current = NO_PENDING_QUIET_UPDATE;
    setGraph((current) => {
      undoStack.current.push(current);
      syncFlags();
      return next;
    });
  }, [syncFlags]);

  const setQuiet = useCallback((next: GraphDefinition | null) => {
    setGraph((prev) => {
      if (quietBaselineRef.current === NO_PENDING_QUIET_UPDATE) {
        quietBaselineRef.current = prev;
      }
      return next;
    });
  }, []);

  const reset = useCallback(
    (next: GraphDefinition | null) => {
      undoStack.current = [];
      redoStack.current = [];
      quietBaselineRef.current = NO_PENDING_QUIET_UPDATE;
      setGraph(next);
      syncFlags();
    },
    [syncFlags],
  );

  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      const isMod = e.metaKey || e.ctrlKey;
      if (!isMod || e.key.toLowerCase() !== "z") return;

      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable)) {
        return;
      }

      e.preventDefault();
      if (e.shiftKey) {
        redo();
      } else {
        undo();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo]);

  return { graph, set, setQuiet, undo, redo, canUndo, canRedo, reset };
}
