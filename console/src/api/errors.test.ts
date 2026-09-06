import { describe, expect, it } from "vitest";
import { describeToolError, RECOVERY_GUIDANCE } from "./errors";
import type { ToolError } from "./types";

describe("error recovery guidance", () => {
  it("covers every error code with guidance and a correlation id in the message", () => {
    const codes = Object.keys(RECOVERY_GUIDANCE) as Array<keyof typeof RECOVERY_GUIDANCE>;
    expect(codes).toContain("OUTCOME_UNKNOWN");
    for (const code of codes) {
      const error: ToolError = { code, message: "boom", correlationId: "corr-1", retry: "never" };
      const text = describeToolError(error);
      expect(text).toContain("corr-1");
      expect(text).toContain(RECOVERY_GUIDANCE[code]);
    }
  });
});
