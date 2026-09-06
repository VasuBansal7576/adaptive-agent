import { describe, expect, it } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "./App";
import { createSimulationTransport } from "./api/simulation";

describe("dbg escape", () => {
  it("escape closes learning dialog", async () => {
    const user = userEvent.setup();
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1004/ });
    await user.click(screen.getAllByRole("button", { name: "New run" })[0]);
    const dialog = await screen.findByRole("dialog", { name: "Create run" });
    let docEvents = 0;
    document.addEventListener("keydown", () => { docEvents += 1; });
    const active = document.activeElement?.tagName;
    await user.keyboard("{Escape}");
    if (docEvents === 0) throw new Error("UE-DEBUG " + JSON.stringify({ docEvents, active }));
    const { fireEvent } = await import("@testing-library/react");
    fireEvent.keyDown(document, { key: "Escape" });
    await new Promise((r) => setTimeout(r, 100));
    console.log("AFTER_SYNTHETIC:", screen.queryByRole("dialog") === null ? "closed" : "open");
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument(), { timeout: 2000 });
    expect(true).toBe(true);
  });
});
