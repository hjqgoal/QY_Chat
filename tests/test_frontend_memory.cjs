const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const staticDir = path.join(__dirname, "..", "chat_reply_agent", "static");
for (const filename of fs.readdirSync(staticDir).filter(name => name.endsWith(".html"))) {
  const html = fs.readFileSync(path.join(staticDir, filename), "utf8");
  for (const match of html.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)) {
    new vm.Script(match[1], { filename });
  }
}
const html = fs.readFileSync(path.join(staticDir, "index.html"), "utf8");
const start = html.indexOf("      function memoryStatusText(");
const end = html.indexOf("      async function submitReplyFeedback(", start);
assert(start > 0 && end > start);
const code = html.slice(start, end);
const weatherStart = html.indexOf("      function renderWeatherEvidence(");
const weatherEnd = html.indexOf("      function renderDiagnostics(", weatherStart);
const escapeStart = html.indexOf("      function escapeHtml(");
const escapeEnd = html.indexOf("      function formatDate(", escapeStart);
const weatherContext = vm.createContext({});
vm.runInContext(html.slice(escapeStart, escapeEnd) + html.slice(weatherStart, weatherEnd), weatherContext);
assert.equal(weatherContext.renderWeatherEvidence([]), "");
const weatherHtml = weatherContext.renderWeatherEvidence([{result: {
  status: "partial", location: {name: "<img src=x onerror=alert(1)>"}, forecast_date: "2026-10-10", period: "evening",
  period_ranges: {temperature_2m: {min: 0, max: 10}}, air_quality: null,
}}]);
assert(!weatherHtml.includes("<img"));
assert(weatherHtml.includes("&lt;img"));
assert(weatherHtml.includes("0 ~ 10 °C"));
assert(weatherHtml.includes("PM2.5：未获取"));
assert(weatherHtml.includes("部分数据可用"));
assert(weatherHtml.includes("CAMS"));

function harness() {
  const elements = new Map();
  const buttons = [{ disabled: true }, { disabled: true }];
  const draft = { value: "user edited reply" };
  const state = { memory_status: "pending", memory_job_id: "job", user_id: "owner", style_evidence: { current_user_message_count: 1 } };
  const sandbox = {
    latestResponse: state, memoryPollToken: 1,
    result: {
      querySelector(selector) {
        if (!elements.has(selector)) elements.set(selector, { textContent: "untouched" });
        return elements.get(selector);
      },
      querySelectorAll: () => buttons,
    },
    document: { getElementById: () => ({ remove() {} }) },
    setTimeout(resolve) { resolve(); },
    loadContacts() {},
    fetch() { throw new Error("Unexpected network call"); },
  };
  vm.createContext(sandbox);
  vm.runInContext(code, sandbox);
  return { sandbox, elements, buttons, draft, state };
}

(async () => {
  let h = harness();
  h.sandbox.updateMemoryStatus(h.state, { status: "completed", memory_persisted: true, saved_memory_id: 12, memory_summary: "summary", style_summary: "style" });
  assert.equal(h.state.saved_memory_id, 12);
  assert(h.buttons.every(button => !button.disabled));
  assert.equal(h.elements.get("[data-memory-summary]").textContent, "summary");
  assert.equal(h.draft.value, "user edited reply");

  h = harness();
  h.sandbox.fetch = async url => {
    assert(url.includes("user_id=owner"));
    h.sandbox.latestResponse = {};
    return { ok: true, status: 200, json: async () => ({ status: "completed", memory_persisted: true }) };
  };
  await h.sandbox.pollMemory(h.state, 1);
  assert.equal(h.elements.size, 0, "Stale responses must not update a different result");

  h = harness();
  h.sandbox.fetch = async () => {
    h.state.memory_status = "cancelled";
    return { ok: true, status: 200, json: async () => ({ status: "running", memory_persisted: false }) };
  };
  await h.sandbox.pollMemory(h.state, 1);
  assert.equal(h.state.memory_status, "cancelled", "An in-flight poll must not undo cancellation");

  h = harness();
  h.sandbox.fetch = async () => ({ ok: false, status: 404 });
  await h.sandbox.pollMemory(h.state, 1);
  assert.equal(h.state.memory_status, "cancelled");

  h = harness();
  h.sandbox.fetch = async () => { throw new Error("offline"); };
  await h.sandbox.pollMemory(h.state, 1);
  assert.equal(h.state.memory_status, "pending", "A polling failure is not a persistence failure");
  assert(h.elements.get("[data-memory-status]").textContent.includes("暂未确认"));
  console.log("Frontend syntax, 5 memory-status checks and weather rendering checks passed.");
})().catch(error => { console.error(error); process.exitCode = 1; });
