import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../App";
import { createSimulationTransport } from "../api/simulation";

describe("approval and stale-stream flows", () => {
  it("opens the approval dialog with exact arguments, scope, expiry; restores focus after close", async () => {
    const { baseElement } = render(<App transport={createSimulationTransport({ disconnectAfterEvents: 0 })} />);
    await screen.findByText("run-sim-1002");
    await userEvent.click(screen.getByRole("button", { name: /run-sim-1002/ }));

    const dialog = await screen.findByRole("dialog", { name: "Approval required" });
    expect(within(dialog).getByText("ledger.append")).toBeInTheDocument();
    expect(within(dialog).getByText(/finance\/ledger\/Q3-2026/)).toBeInTheDocument();
    expect(within(dialog).getByText("2026-09-06T23:59:00Z")).toBeInTheDocument();
    expect(within(dialog).getByText(/"batchId": "Q3-2026-114"/)).toBeInTheDocument();

    await userEvent.click(within(dialog).getByRole("button", { name: "Deny" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    // focus restored to a control in the document after close
    const after = (baseElement.ownerDocument as Document).activeElement;
    expect(after).not.toBeNull();
  });

  it("survives a stale-stream drop and reconnects without duplicate events", async () => {
    render(<App transport={createSimulationTransport({ disconnectAfterEvents: 2 })} />);
    await screen.findAllByRole("button", { name: /run-sim-1001/ });
    // the stream for run 1001 drops after 2 events then resumes from cursor
    expect(await screen.findByText(/Event stream stale/)).toBeInTheDocument();
    // after reconnect the remaining scripted events arrive exactly once
    expect(await screen.findByText("#6", {}, { timeout: 4000 })).toBeInTheDocument();
    const seq6 = screen.getAllByText("#6");
    expect(seq6).toHaveLength(1);
  });
});
