const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "..", "chat_reply_agent/static/index.html"), "utf8");
const source = html.slice(html.indexOf("      function updateSingleQuality("), html.indexOf("      function renderWeatherEvidence("));
const escaping = html.slice(html.indexOf("      function escapeHtml("), html.indexOf("      function formatDate("));
const texts = ["行，那我先去了", "好啦，你们玩开心", "嗯，下次再说"];
const element = () => ({textContent: "", innerHTML: "", disabled: false, hidden: false, classList: {toggle() {}}});

function fixture() {
  const inputs = texts.map(value => ({value, selectionStart: 2, selectionEnd: 2}));
  const badges = texts.map(element);
  const details = texts.map(element);
  const selectors = Object.fromEntries(["total", "summary", "state", "group", "job-state", "diagnostics", "recommendation"].map(key => [`[data-quality-${key}]`, element()]));
  selectors["[data-action='review-current']"] = element();
  selectors["[data-action='cancel-quality']"] = element();
  const cards = texts.map((_, index) => ({dataset: {replyIndex: String(index)}, querySelector(selector) {
    return {"[data-reply-text]": inputs[index], ".quality-badge": badges[index], "[data-quality-details]": details[index]}[selector];
  }}));
  const result = {
    querySelectorAll: selector => selector === "[data-reply-index]" ? cards : inputs,
    querySelector(selector) {
      const match = selector.match(/data-reply-index="(\d)"/);
      return match ? inputs[Number(match[1])] : selectors[selector];
    },
  };
  const data = {user_id: "owner", quality_job_id: "job1", quality_status: "pending", replies: texts.map(text => ({text})),
    quality: {status: "pending", passed: false, items: texts.map((_, candidate_index) => ({candidate_index, status: "pending", verdict: "unverified"}))}};
  const sandbox = vm.createContext({result, latestResponse: data, qualityPollToken: 1, replyRequestToken: 0,
    contextNeedsRegeneration: false, setTimeout: fn => fn(), fetch: () => {throw Error("unexpected fetch");}});
  vm.runInContext(escaping + source, sandbox);
  return {sandbox, data, inputs, badges, details, selectors};
}

function completed(jobId = "job1", originals = texts) {
  return {job_id: jobId, status: "completed", candidate_texts: originals,
    diagnostics: {total_ms: 12000, queue_wait_ms: 400},
    quality: {status: "reviewed", passed: true, diversity_status: "distinct", group_issues: [],
      recommended_candidate_indexes: [0], recommendation_reason: "保留原有时间限制", items: originals.map((text, candidate_index) => ({
      candidate_index, evaluated_text: text, status: "reviewed", verdict: "usable", passed: true, assessment_source: "read_only",
      evidence: {naturalness: "<img src=x onerror=alert(1)>"}, naturalness: 4,
    }))}};
}

const response = payload => ({ok: true, status: 200, json: async () => payload});
function deferred() {
  let resolve;
  const promise = new Promise(r => {resolve = r;});
  return {promise, resolve};
}

(async () => {
  {
    const f = fixture();
    let calls = 0;
    f.sandbox.fetch = async () => response(++calls === 1 ? {job_id: "job1", status: "running"} : completed());
    await f.sandbox.pollQuality(f.data, 1, "job1", texts);
    assert.equal(calls, 2);
    assert.equal(f.data.quality_status, "completed");
    assert.deepEqual(f.inputs.map(input => input.value), texts);
    assert(f.inputs.every(input => input.selectionStart === 2));
    assert(f.badges.every(badge => badge.textContent === "基本可用"));
    assert(f.details[0].innerHTML.includes("&lt;img"));
    assert(!f.details[0].innerHTML.includes("<img"));
    assert(f.selectors["[data-quality-diagnostics]"].innerHTML.includes("12.00 秒"));
    assert(f.selectors["[data-quality-recommendation]"].textContent.includes("候选 1"));
  }
  {
    const f = fixture();
    f.inputs[1].value = "这是我自己修改的";
    f.inputs[1].selectionStart = 5;
    assert(f.sandbox.applyQualityResult(f.data, completed(), texts));
    assert.equal(f.inputs[1].value, "这是我自己修改的");
    assert.equal(f.inputs[1].selectionStart, 5);
    assert.equal(f.badges[1].textContent, "已编辑，待确认");
    assert.equal(f.badges[0].textContent, "基本可用");
    assert.equal(f.data.quality.passed, false);
    assert.equal(f.data.quality.diversity_status, "not_reviewed");
    assert.equal(f.data.quality.recommended_candidate_indexes.length, 0);
    assert.equal(f.selectors["[data-quality-recommendation]"].textContent, "");
  }
  {
    const f = fixture();
    const partial = completed();
    partial.quality.status = "partial";
    partial.quality.passed = false;
    partial.quality.items[1] = {candidate_index: 1, status: "invalid_response", assessment_source: "local", verdict: "unverified", passed: false};
    partial.message = "已保留 2/3 条有效评价；本轮校验未全部完成";
    partial.diagnostics.validation_errors = [{path: "items.1.review.evidence.relevance", type: "missing_or_blank_evidence"},
      {path: "<img src=x>", type: "<script>"}];
    assert(f.sandbox.applyQualityResult(f.data, partial, texts));
    assert.deepEqual(f.inputs.map(input => input.value), texts);
    assert.equal(f.badges[0].textContent, "基本可用");
    assert.equal(f.badges[1].textContent, "复核未完成");
    assert.equal(f.badges[2].textContent, "基本可用");
    assert.equal(f.selectors["[data-quality-total]"].textContent, "待确认");
    assert(f.selectors["[data-quality-job-state]"].textContent.includes("2/3"));
    const diagnostics = f.selectors["[data-quality-diagnostics]"].innerHTML;
    assert(diagnostics.includes("本维度缺少评价依据"));
    assert(diagnostics.includes("&lt;img"));
    assert(!diagnostics.includes("<img"));
  }
  {
    const f = fixture();
    // Even an edit restored to the old spelling requires an explicit new review.
    f.data.quality_edited_indexes = [0];
    assert(f.sandbox.applyQualityResult(f.data, completed(), texts));
    assert.equal(f.badges[0].textContent, "已编辑，待确认");
    assert.equal(f.sandbox.applyQualityResult(f.data, completed("old-job"), texts), false);
    assert.equal(f.sandbox.applyQualityResult(f.data, completed("job1", ["wrong", ...texts.slice(1)]), texts), false);
  }
  for (const phase of ["fetch", "json"]) {
    const f = fixture();
    const pending = deferred();
    f.sandbox.fetch = phase === "fetch" ? () => pending.promise : async () => ({ok: true, status: 200, json: () => pending.promise});
    const task = f.sandbox.pollQuality(f.data, 1, "job1", texts);
    await Promise.resolve();
    f.sandbox.qualityPollToken++;
    f.sandbox.latestResponse = {quality_status: "pending", replies: [{text: "新一轮"}]};
    pending.resolve(phase === "fetch" ? response(completed()) : completed());
    await task;
    assert.equal(f.data.quality_status, "pending");
    assert.equal(f.sandbox.latestResponse.replies[0].text, "新一轮");
    assert(f.badges.every(badge => badge.textContent === ""));
  }
  for (const state of ["failed", "timed_out", "cancelled"]) {
    const f = fixture();
    f.sandbox.fetch = async () => response({job_id: "job1", status: state});
    await f.sandbox.pollQuality(f.data, 1, "job1", texts);
    assert.equal(f.data.quality_status, state);
    assert.equal(f.data.quality.passed, false);
    assert(f.badges.every(badge => badge.textContent !== "基本可用"));
    assert.equal(f.selectors["[data-action='review-current']"].disabled, false);
    assert.equal(f.selectors["[data-action='cancel-quality']"].hidden, true);
  }
  {
    const f = fixture();
    f.sandbox.fetch = async () => ({status: 404, ok: false});
    await f.sandbox.pollQuality(f.data, 1, "job1", texts);
    assert.equal(f.data.quality_status, "unavailable");
    assert(f.data.quality.summary.includes("重启"));
  }
  {
    const f = fixture();
    let calls = 0;
    f.sandbox.fetch = async () => {calls++; throw Error("offline");};
    await f.sandbox.pollQuality(f.data, 1, "job1", texts);
    assert.equal(calls, 3);
    assert.equal(f.data.quality_status, "unavailable");
    assert.equal(f.data.quality.passed, false);
  }
  {
    const f = fixture();
    const pending = deferred();
    let deletes = 0;
    f.sandbox.fetch = (_, options) => options?.method === "DELETE" ? (deletes++, Promise.resolve({})) : pending.promise;
    const task = f.sandbox.pollQuality(f.data, 1, "job1", texts);
    f.sandbox.stopQualityPolling(f.data, "已取消");
    pending.resolve(response(completed()));
    await task;
    assert.equal(deletes, 1);
    assert.equal(f.data.quality_status, "cancelled");
    assert.equal(f.data.quality.passed, false);
    assert.deepEqual(f.inputs.map(input => input.value), texts);
  }
  {
    const f = fixture();
    const pending = deferred();
    f.sandbox.syncEditedReplyTexts = () => {};
    f.sandbox.collectReplyControls = () => ({});
    let submitted;
    f.sandbox.fetch = (_, options) => {
      if (options?.method === "DELETE") return Promise.resolve({});
      submitted = JSON.parse(options.body);
      return pending.promise;
    };
    // Stop actual polling so this test can isolate edits during review submission.
    f.sandbox.startQualityPolling = () => {};
    const task = f.sandbox.reviewCurrentReplies(f.selectors["[data-action='review-current']"]);
    f.inputs[2].value = "提交评估时又改了一句";
    pending.resolve(response({job_id: "job2", status: "pending", candidate_texts: texts, quality: f.data.quality}));
    await task;
    assert.equal(submitted.save_to_memory, false);
    assert.equal(f.data.quality_job_id, "job2");
    assert(f.data.quality_edited_indexes.includes(2));
    assert(f.sandbox.applyQualityResult(f.data, completed("job2"), texts));
    assert.equal(f.inputs[2].value, "提交评估时又改了一句");
    assert.equal(f.badges[2].textContent, "已编辑，待确认");
  }
  console.log("Async review polling, read-only updates, edits, stale results, cancellation, retry and failure checks passed.");
})().catch(error => {console.error(error); process.exitCode = 1;});
