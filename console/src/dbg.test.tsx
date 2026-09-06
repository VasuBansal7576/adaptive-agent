import { describe, expect, it } from "vitest";
import { normalizeSseEvent } from "./api/validate";

describe("dbg wire", () => {
  it("qa captured payload", () => {
    const wire = {
      id: 1,
      event: "run_created",
      data: {
        evidence_id: "ev_370205e6caa74d32b05b04c2377deab0",
        run_id: "run_873a94b0ce424e7699709663b461eb13",
        sequence: 1,
        event_type: "run_created",
        content_hash: "9baec1f5358fbce8fa7d8a27fc4be5cb89d28857c67daf0b7ed83cdf8fe610b5",
        source_ref: '{"id":"art_9baec1f5358fbce8","version":"1","sha256":"9baec1f5358fbce8fa7d8a27fc4be5cb89d28857c67daf0b7ed83cdf8fe610b5"}',
        trust_class: "system",
        visibility: "learner",
        redacted: 1,
      },
    };
    const e = normalizeSseEvent(wire, "sse");
    console.log("PARSED:", JSON.stringify(e));
    expect(e.sequence).toBe(1);
  });
});
