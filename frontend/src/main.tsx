import ReactDOM from "react-dom/client";

import App from "./App";
import "./styles.css";

function isBenignResizeObserverMessage(message: string): boolean {
  return message === "ResizeObserver loop completed with undelivered notifications.";
}

window.addEventListener("error", (event) => {
  if (isBenignResizeObserverMessage(event.message)) {
    event.preventDefault();
    event.stopImmediatePropagation();
  }
});

window.addEventListener("unhandledrejection", (event) => {
  if (
    event.reason instanceof Error &&
    isBenignResizeObserverMessage(event.reason.message)
  ) {
    event.preventDefault();
  }
});

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(<App />);
