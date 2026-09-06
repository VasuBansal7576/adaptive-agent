import puppeteer from "puppeteer-core";
const browser = await puppeteer.launch({ executablePath: "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless: "new", args: ["--no-sandbox"] });
const page = await browser.newPage();
await page.setViewport({ width: 375, height: 1000 });
await page.goto("http://127.0.0.1:8910/", { waitUntil: "networkidle2" });
await new Promise((r) => setTimeout(r, 1200));
const badge = await page.evaluate(() => {
  const pills = [...document.querySelectorAll("span")].filter((el) => (el.className || "").includes("rounded-full"));
  return pills.slice(0, 3).map((el) => {
    const r = el.getBoundingClientRect();
    return { text: el.textContent, left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width), cardRight: Math.round(el.closest("button")?.getBoundingClientRect().right ?? -1) };
  });
});
const firstControl = await page.evaluate(() => {
  const el = document.querySelector("body button, body a");
  return el ? (el.getAttribute("aria-label") ?? el.textContent ?? "").slice(0, 40) : "none";
});
console.log("badges:", JSON.stringify(badge));
console.log("firstControl:", firstControl);
await page.screenshot({ path: "/Users/vasu/.ao/data/worktrees/adaptive-agent/adaptive-agent-5/console/qa/acc010/probe-375.png" });
await browser.close();
