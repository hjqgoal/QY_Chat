const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "..", "chat_reply_agent", "static", "index.html"), "utf8");
const start = html.indexOf("      function updateSingleQuality(");
const end = html.indexOf("      function renderWeatherEvidence(", start);
const escapeStart = html.indexOf("      function escapeHtml(");
const escapeEnd = html.indexOf("      function formatDate(", escapeStart);
assert(start > 0 && end > start);
const sandbox = vm.createContext({latestResponse: {}});
vm.runInContext(html.slice(escapeStart, escapeEnd) + html.slice(start, end), sandbox);
const review = {
  candidate_index: 0, status: "reviewed", assessment_source: "targeted_review", verdict: "usable",
  evaluated_text: "那你先试试", passed: true, overall_score: null, naturalness: 4, style_fit: null,
  evidence: {naturalness: "<img src=x onerror=alert(1)>", style_fit: "本人样本不足"},
  conclusion: "<script>bad()</script>这句能接住话",
  advisories: ["不保证对方反应"], uncertainties: ["需要确认时间"],
};
assert.equal(sandbox.qualityLabel(review, review.evaluated_text), "基本可用");
assert.equal(sandbox.qualityLabel({...review, verdict: "uncertain"}, review.evaluated_text), "信息不足");
assert.equal(sandbox.qualityLabel({...review, verdict: "needs_attention"}, review.evaluated_text), "建议调整");
assert.equal(sandbox.qualityLabel({...review, status: "revised_pending"}, review.evaluated_text), "已修改，待确认");
assert.equal(sandbox.qualityLabel(review, "我改过的文字"), "文本已变，待确认");
assert(!sandbox.renderQualityDetails(review, "我改过的文字").includes("本人样本不足"));
const details = sandbox.renderQualityDetails(review, review.evaluated_text);
assert(details.includes("&lt;img"));
assert(!details.includes("<img"));
assert(details.includes("本人风格（未确认）"));
assert(details.includes("4/5 · 基本贴切"));
assert(details.includes("暂不计分"));
assert(!details.includes("0/5"));
assert(details.includes("需补充的关键信息"));
assert(details.includes("普通提醒（不阻止使用）"));
assert(details.includes("&lt;script&gt;"));
assert(!details.includes("<script>"));

const legacy = {candidate_index: 0, status: "reviewed", overall_score: 100, passed: true};
assert.equal(sandbox.qualityLabel(legacy, "旧回复"), "历史评价，未校准");
assert.equal(sandbox.qualityLabel({...legacy, assessment_source: "legacy"}, "旧回复"), "历史评价，未校准");
assert(!sandbox.qualitySummary({items: [legacy], summary: "平均100分"}).includes("100"));
assert.equal(sandbox.qualityGroupLabel({items: [legacy], passed: true}, [{text: "旧回复"}]), "历史评价，未校准");
assert.equal(sandbox.qualityLabel({...review, status: "unavailable"}, review.evaluated_text), "复核未完成");

const items = Array.from({length: 3}, (_, candidate_index) => ({...review, candidate_index}));
sandbox.latestResponse = {quality: {items, passed: true, average_score: 100, diversity_status: "distinct",
  recommended_candidate_indexes: [0], recommendation_reason: "旧建议", group_reason: "旧差异"}};
sandbox.updateSingleQuality({...review, candidate_index: 1, status: "not_reviewed", verdict: "unverified", evidence: {}, passed: false}, 1);
assert.equal(sandbox.latestResponse.quality.average_score, null);
assert.equal(sandbox.latestResponse.quality.diversity_status, "not_reviewed");
assert.equal(sandbox.latestResponse.quality.passed, false);
assert.equal(sandbox.latestResponse.quality.items[0].verdict, "usable");
assert.equal(sandbox.latestResponse.quality.items[1].status, "not_reviewed");
assert.equal(sandbox.latestResponse.quality.recommended_candidate_indexes.length, 0);
assert.equal(sandbox.latestResponse.quality.recommendation_reason, "");
assert.equal(sandbox.latestResponse.quality.group_reason, "");
const recommendation = {status: "reviewed", items, recommended_candidate_indexes: [0], recommendation_reason: "贴切"};
const replies = items.map(item => ({text: item.evaluated_text}));
assert(sandbox.qualityRecommendation(recommendation, replies).includes("候选 1"));
assert.equal(sandbox.qualityRecommendation({...recommendation, status: "partial"}, replies), "");
assert.equal(sandbox.qualityRecommendation(recommendation, [{text: "edited"}, ...replies.slice(1)]), "");
assert.equal(sandbox.qualityGroupLabel({items, passed: false, diversity_status: "similar"}, []), "候选差异不足");
console.log("Frontend quality labels, legacy scores, evidence escaping and invalidation checks passed.");
