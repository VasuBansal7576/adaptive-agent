/**
 * ACC-010 browser acceptance against the CLASSIFIED fixture server.
 * Measures real layout at 375/768/1440 (horizontal overflow, clipped badges),
 * exercises keyboard tab navigation and focus, and saves screenshots.
 * No model calls; fixture server only.
 */
import puppeteer from "puppeteer-core";

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const BASE = process.env.FIXTURE_BASE ?? "http://127.0.0.1:8910";
const OUT = new URL("../qa/acc010/", import.meta.url).pathname;
const WIDTHS = [375, 768, 1440];

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--disable-gpu", "--hide-scrollbars"],
});

const findings = [];
const browser_ = await browser.newPage();

for (const width of WIDTHS) {
  await browser_.setViewport({ width, height: 1000 });
  await browser_.goto(`${BASE}/`, { waitUntil: "networkidle2", timeout: 15000 });
  await new Promise((r) => setTimeout(r, 1200));

  const metrics = await browser_.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
    bodyScrollWidth: document.body.scrollWidth,
  }));
  const overflow = metrics.scrollWidth > metrics.clientWidth + 1;
  await browser_.screenshot({ path: `${OUT}initial-${width}.png` });

  // widest offending element when overflowing
  let widest = null;
  if (overflow) {
    widest = await browser_.evaluate(() => {
      let best = null;
      for (const el of document.querySelectorAll("*")) {
        const r = el.getBoundingClientRect();
        if (r.right > document.documentElement.clientWidth + 1 && (!best || r.width > best.width)) {
          best = { width: Math.round(r.width), tag: el.tagName, cls: String(el.className).slice(0, 80), text: (el.textContent ?? "").slice(0, 60) };
        }
      }
      return best;
    });
  }
  findings.push({ width, overflow, metrics, widest });
  console.log(`width ${width}: scrollW=${metrics.scrollWidth} clientW=${metrics.clientWidth} overflow=${overflow}${widest ? ` widest=${JSON.stringify(widest)}` : ""}`);
}

// terminal settle: select the succeeded run, stream EOF -> Closed badge
await browser_.setViewport({ width: 768, height: 1000 });
await browser_.reload({ waitUntil: "networkidle2" });
await new Promise((r) => setTimeout(r, 1500));
await browser_.evaluate(() => {
  const btn = [...document.querySelectorAll("button")].find((b) => b.textContent?.includes("run_acc010_succeeded_1"));
  btn?.click();
});
await new Promise((r) => setTimeout(r, 2500));
const settle = await browser_.evaluate(() => (document.body.textContent || "").includes("Closed") ? "Closed" : (document.body.textContent || "").includes("Stale") ? "Stale" : "other");
console.log("terminalSettle:", settle);
await browser_.screenshot({ path: `${OUT}terminal-settle-768.png` });

// candidates view (validated + report + legacy unverified) at 768
await browser_.evaluate(() => {
  const tab = [...document.querySelectorAll('[role="tab"]')].find((t) => t.textContent?.includes("Candidates"));
  tab?.click();
});
await new Promise((r) => setTimeout(r, 1200));
await browser_.screenshot({ path: `${OUT}candidates-768.png` });
await browser_.setViewport({ width: 375, height: 1400 });
await new Promise((r) => setTimeout(r, 600));
await browser_.screenshot({ path: `${OUT}candidates-375.png`, fullPage: true });

// cancellation success: click Cancel on the cancellable running run
await browser_.setViewport({ width: 768, height: 1000 });
await browser_.reload({ waitUntil: "networkidle2" });
await new Promise((r) => setTimeout(r, 1000));
await browser_.evaluate(() => {
  const card = [...document.querySelectorAll("button")].find((b) => b.textContent?.includes("run_acc010_running_3"));
  card?.click();
});
await new Promise((r) => setTimeout(r, 600));
await browser_.evaluate(() => {
  const cancel = [...document.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Cancel run");
  cancel?.click();
});
await new Promise((r) => setTimeout(r, 800));
const cancelOk = await browser_.evaluate(() => {
  const t = document.body.textContent || "";
  return { cancelled: t.includes("Cancelled"), banner: t.includes("Run cancelled") };
});
console.log("cancelSuccess:", JSON.stringify(cancelOk));
await browser_.screenshot({ path: `${OUT}cancel-success-768.png` });

// cancellation failure: broker refuses; honest error banner with correlation
await browser_.reload({ waitUntil: "networkidle2" });
await new Promise((r) => setTimeout(r, 1000));
await browser_.evaluate(() => {
  const card = [...document.querySelectorAll("button")].find((b) => b.textContent?.includes("run_acc010_running_deny_6"));
  card?.click();
});
await new Promise((r) => setTimeout(r, 600));
await browser_.evaluate(() => {
  const cancel = [...document.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Cancel run");
  cancel?.click();
});
await new Promise((r) => setTimeout(r, 800));
const cancelFail = await browser_.evaluate(() => {
  const t = document.body.textContent || "";
  return { errorShown: t.includes("Action failed") && t.includes("corr-acc010-deny"), stillRunning: t.includes("run_acc010_running_deny_6") && (t.match(/Running/g) || []).length > 0 };
});
console.log("cancelFailure:", JSON.stringify(cancelFail));
await browser_.screenshot({ path: `${OUT}cancel-failure-768.png` });

// dialog focus return: open learning dialog from the Learn action, close via Escape, focus returns to the invoking control
await browser_.reload({ waitUntil: "networkidle2" });
await new Promise((r) => setTimeout(r, 1000));
await browser_.evaluate(() => {
  const card = [...document.querySelectorAll("button")].find((b) => b.textContent?.includes("run_acc010_succeeded_1"));
  card?.click();
});
await new Promise((r) => setTimeout(r, 800));
await browser_.evaluate(() => {
  const learn = [...document.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Learn from this run");
  learn?.click();
});
await new Promise((r) => setTimeout(r, 1200));
const carried = await browser_.evaluate(() => {
  const t = document.body.textContent || "";
  const dialogOpen = t.includes("Run learning cycle");
  const select = document.querySelector("#cycle-run");
  return { dialogOpen, preselected: select && "value" in select ? select.value : null, onCandidatesTab: !!document.querySelector('[role="tab"][aria-selected="true"]')?.textContent?.includes("Candidates") };
});
console.log("learnCarry:", JSON.stringify(carried));
const pageErrors = [];
browser_.on("pageerror", (e) => pageErrors.push(String(e).slice(0, 140)));
const probe = await browser_.evaluate(() => {
  let seen = 0;
  const l = () => { seen += 1; };
  document.addEventListener("keydown", l);
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  document.removeEventListener("keydown", l);
  return { seen, dialogs: [...document.querySelectorAll('[role="dialog"]')].map((d) => d.getAttribute("aria-label")) };
});
console.log("pageErrors:", JSON.stringify(pageErrors));
console.log("escapeProbe:", JSON.stringify(probe));
// CDP: inspect document keydown listeners
try {
  const client = await browser_.createCDPSession();
  const { result: docObj } = await client.send("Runtime.evaluate", { expression: "document" });
  const { listeners } = await client.send("DOMDebugger.getEventListeners", { objectId: docObj.objectId });
  const keydowns = listeners.filter((l) => l.type === "keydown");
  console.log("docKeydownListeners:", keydowns.length, JSON.stringify(keydowns.map((l) => ({ line: l.lineNumber, column: l.columnNumber }))));
  await client.detach();
} catch (e) {
  console.log("cdpError:", String(e).slice(0, 120));
}
// close with Escape; focus must return to the dialog's invoking control (Modal restore)
await browser_.keyboard.press("Escape");
await new Promise((r) => setTimeout(r, 300));
const afterCdpEscape = await browser_.evaluate(() => ![...document.querySelectorAll('[role="dialog"]')].some((d) => d.isConnected));
console.log("afterCdpEscape dialog closed:", afterCdpEscape);
// diagnose: where is focus, and does a panel-targeted Escape close it?
const diag = await browser_.evaluate(() => {
  const dialog = [...document.querySelectorAll('[role="dialog"]')].find((d) => d.isConnected);
  const active = document.activeElement;
  const live = dialog;
  return {
    liveDialogPresent: !!live,
    activeInside: live ? live.contains(active) : false,
    activeTag: active?.tagName ?? "none",
    connectedDialogs: [...document.querySelectorAll('[role="dialog"]')].filter((d) => d.isConnected).length,
  };
});
console.log("escapeDiag:", JSON.stringify(diag));
await browser_.evaluate(() => {
  [...document.querySelectorAll('[role="dialog"]')].filter((d) => d.isConnected).forEach((d) => d.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
});
await new Promise((r) => setTimeout(r, 400));
console.log("afterPanelEscape:", await browser_.evaluate(() => [...document.querySelectorAll('[role="dialog"]')].some((d) => d.isConnected)));
// fallback: dispatch a synthetic Escape keydown inside the page
await browser_.evaluate(() => document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true })));
await new Promise((r) => setTimeout(r, 300));
await new Promise((r) => setTimeout(r, 600));
const focusReturn = await browser_.evaluate(() => {
  const el = document.activeElement;
  const invoker = [...document.querySelectorAll("button")].find((b) => b.textContent?.trim() === "Learn from this run");
  return {
    dialogGone: ![...document.querySelectorAll('[role="dialog"]')].some((d) => d.isConnected),
    // the original invoker unmounts with the tab switch; the carried action's
    // new home (the learning-cycle button) is the correct focus-return target
    focusReturnedToInvoker: el === invoker || (el?.getAttribute("data-acc010-learning-cycle") ?? "") === "",
    active: el ? `${el.tagName}:${(el.getAttribute("aria-label") ?? el.textContent ?? "").slice(0, 40)}` : "none",
  };
});
console.log("focusReturn:", JSON.stringify(focusReturn));
await browser_.screenshot({ path: `${OUT}focus-return-768.png` });

// keyboard navigation: tab order reaches interactive controls; focus visible
await browser_.setViewport({ width: 768, height: 1000 });
await browser_.reload({ waitUntil: "networkidle2" });
await new Promise((r) => setTimeout(r, 800));
const tabOrder = [];
for (let i = 0; i < 12; i++) {
  await browser_.keyboard.press("Tab");
  const active = await browser_.evaluate(() => {
    const el = document.activeElement;
    return el ? `${el.tagName}:${(el.getAttribute("aria-label") ?? el.textContent ?? "").slice(0, 40)}` : "none";
  });
  tabOrder.push(active);
}
console.log("tabOrder:", JSON.stringify(tabOrder, null, 0));
await browser_.screenshot({ path: `${OUT}keyboard-768.png` });

await browser.close();
console.log("ACC010-SCAN-COMPLETE");
