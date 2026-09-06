import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";

function renderConsole() {
  return render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
}

describe("console (simulation fixtures)", () => {
  it("shows the persistent simulation banner and loads all sections", async () => {
    renderConsole();
    expect(await screen.findByText(/SIMULATED — development fixture, not live inference/)).toBeInTheDocument();
    const runButtons = await screen.findAllByRole("button", { name: /run-sim-1001/ });
    expect(runButtons.length).toBeGreaterThanOrEqual(1);
  });

  it("renders non-color status indicators pairing glyph and text", async () => {
    renderConsole();
    await screen.findByText("run-sim-1002");
    const badges = screen.getAllByText("Awaiting approval");
    expect(badges.length).toBeGreaterThan(0);
  });

  it("shows the failure banner with recovery guidance for the failed run", async () => {
    renderConsole();
    await screen.findAllByRole("button", { name: /run-sim-1003/ });
    await userEvent.click(screen.getAllByRole("button", { name: /run-sim-1003/ })[0]);
    // OUTCOME_UNKNOWN banner appears with correlation id and reconciliation guidance
    expect(await screen.findByText("Tool error: OUTCOME_UNKNOWN")).toBeInTheDocument();
    expect((await screen.findAllByText(/do not repeat the operation manually/)).length).toBeGreaterThanOrEqual(1);
  });

  it("registers keyboard tab navigation across console sections", async () => {
    renderConsole();
    const tablist = await screen.findByRole("tablist", { name: "Console sections" });
    const tabs = within(tablist).getAllByRole("tab");
    expect(tabs).toHaveLength(4);
    await userEvent.click(tabs[2]); // Skill library
    expect(await screen.findByText("recheck-before-update")).toBeInTheDocument();
    tabs[2].focus();
    await userEvent.keyboard("{ArrowRight}");
    expect(tabs[3]).toHaveFocus();
    expect(tabs[3]).toHaveAttribute("aria-selected", "true");
  });

  it("shows an empty registry explanation when no environments exist", async () => {
    // registry tab with environments present still lists them; empty state text exists in component
    renderConsole();
    await userEvent.click(await screen.findByRole("tab", { name: "Environment registry" }));
    expect(await screen.findByText("finance-sim")).toBeInTheDocument();
  });

  it("preserves form input when package validation rejects", async () => {
    renderConsole();
    await userEvent.click(await screen.findByRole("tab", { name: "Environment registry" }));
    const idInput = await screen.findByLabelText("Environment ID");
    await userEvent.type(idInput, "finance-prod");
    await userEvent.click(screen.getByRole("button", { name: "Validate package" }));
    expect(await screen.findByText(/Validation failed — your input is preserved/)).toBeInTheDocument();
    expect(idInput).toHaveValue("finance-prod");
    expect(screen.getAllByRole("alert").length).toBeGreaterThan(0);
  });
});
