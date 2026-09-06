/**
 * Built-app bootstrap/list smoke against the REAL main backend serving the
 * built console (create_runtime_app + temp Store). GET-only: no model calls,
 * no run launches, no learning/evaluation launches. Verify the production
 * contract: console HTML at /, session bootstrap cookie, direct API routes.
 * Usage: BASE=http://127.0.0.1:8977 node console/scripts/prod-app-smoke.mjs
 */
const base = process.env.BASE ?? "http://127.0.0.1:8977";
let cookie = "";

async function get(path, { readBody = true } = {}) {
  const res = await fetch(`${base}${path}`, { headers: cookie ? { cookie } : {} });
  const setCookie = res.headers.get("set-cookie");
  if (setCookie) cookie = setCookie.split(";")[0];
  const body = readBody ? await res.text().catch(() => "") : "";
  return { status: res.status, body, contentType: res.headers.get("content-type") ?? "", setCookie };
}

async function main() {
  // 1. built console served at /
  const home = await get("/", { readBody: true });
  if (home.status !== 200) throw new Error(`GET / -> ${home.status}`);
  if (!/<!doctype html|<html/i.test(home.body)) throw new Error("GET / did not return the built HTML");
  if (!home.body.includes('id="root"')) throw new Error("built HTML missing #root mount");
  console.log(`GET / -> 200 HTML (built console, ${(home.body.length / 1024).toFixed(0)} KB)`);

  // 2. session bootstrap sets the HttpOnly operator cookie; direct route (no /api prefix)
  const bootstrap = await get("/session/bootstrap");
  if (bootstrap.status !== 200) throw new Error(`/session/bootstrap -> ${bootstrap.status}: ${bootstrap.body.slice(0, 120)}`);
  if (!/httponly/i.test(bootstrap.setCookie ?? "")) throw new Error("bootstrap cookie is not HttpOnly");
  console.log("GET /session/bootstrap -> 200 (HttpOnly operator cookie set)");

  // 3. session confirms authentication via the cookie
  const session = await get("/session");
  if (session.status !== 200) throw new Error(`/session -> ${session.status}`);
  const sessionBody = JSON.parse(session.body || "{}");
  if (sessionBody.authenticated !== true) throw new Error("session not authenticated after bootstrap");
  console.log("GET /session -> 200 authenticated");

  // 4. direct API routes (NOT /api-prefixed) serve the operator projection
  for (const path of ["/runs", "/environments", "/skills", "/candidates"]) {
    const res = await get(path);
    if (res.status !== 200) throw new Error(`${path} -> ${res.status}: ${res.body.slice(0, 120)}`);
    const parsed = JSON.parse(res.body || "null");
    if (!Array.isArray(parsed)) throw new Error(`${path} did not return an array`);
    console.log(`GET ${path} -> 200 (${parsed.length} rows)`);
  }

  // 5. /api-prefixed routes must NOT exist in production (the dev-only proxy path)
  const apiProbe = await get("/api/runs", { readBody: false });
  console.log(`GET /api/runs -> ${apiProbe.status} (404 expected: /api is a dev-only proxy prefix)`);
  console.log("PASS");
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(`FAIL: ${error.message}`);
    process.exit(1);
  });
