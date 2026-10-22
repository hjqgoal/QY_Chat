const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const html = fs.readFileSync(path.join(__dirname, "..", "chat_reply_agent", "static", "index.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
function between(start, end) {
  const from = html.indexOf(`      function ${start}(`);
  const to = html.indexOf(`      function ${end}(`, from);
  assert(from > 0 && to > from);
  return html.slice(from, to);
}
const node = {textContent: "old topic"};
const sandbox = vm.createContext({
  structuredClone, pendingExtraction: null, latestResponse: null, contextNeedsRegeneration: false,
  stopQualityPolling() {},
  backgroundInput: {value: "  已约好别人。\r\n\r\n\r\n  只能临时补位。  "},
  result: {querySelectorAll: selector => selector === "[data-topic]" ? [node] : []},
});
vm.runInContext(between("escapeHtml", "formatDate") + between("renderTopicSummary", "buildCorrectedPayload")
  + between("collectEditedChat", "buildChatPayload") + between("getBackgroundContext", "updateSubmitButtonLabel"), sandbox);
assert.equal(sandbox.getBackgroundContext(), "已约好别人。\n\n只能临时补位。");
assert(!sandbox.renderTopicSummary({topic: "去玩了"}).includes("去玩了"));
assert(!sandbox.renderTopicSummary({topic: "去玩了", topic_source: "pending"}).includes("去玩了"));
const rendered = sandbox.renderTopicSummary({topic: "<script>alert(1)</script>", topic_source: "context"});
assert(rendered.includes("&lt;script&gt;"));
assert(!rendered.includes("<script>"));
assert(rendered.includes("模型概括"));
assert(sandbox.renderTopicSummary({topic: "  ", topic_source: "context"}).includes("暂无法概括"));

sandbox.pendingExtraction = {topic: "旧识别主题", topic_source: "context"};
sandbox.latestResponse = {extracted_chat: {topic: "旧生成主题", topic_source: "context"}};
sandbox.invalidateTopic();
assert.equal(sandbox.pendingExtraction.topic, "");
assert.equal(sandbox.latestResponse.extracted_chat.topic_source, "pending");
assert.equal(sandbox.contextNeedsRegeneration, true);
assert(node.textContent.includes("待重新分析"));

const base = {topic: "旧主题", topic_source: "context", messages: [{text: "之前", speaker: "other"}]};
const inputs = {speaker: "me", side: "right", text: "改过的消息"};
sandbox.result.querySelectorAll = () => [{dataset: {messageIndex: "0"}, querySelector: selector => ({
  value: inputs[selector.match(/data-field="(\w+)"/)[1]],
})}];
const edited = sandbox.collectEditedChat(base);
assert.equal(edited.topic, "");
assert.equal(edited.topic_source, "pending");
assert.equal(edited.messages[0].text, "改过的消息");
assert.equal(base.topic, "旧主题");
assert.equal(edited.last_message_from_other, "");

const preview = between("renderExtractionReview", "renderMessageEditors");
assert(!preview.includes("data.topic"));
assert(preview.includes("待结合校对内容与补充背景分析"));
assert(preview.includes("模型估计，非准确率"));
assert(!preview.includes("置信度正常"));
vm.runInContext(between("renderMessageEditors", "renderResult"), sandbox);
const unknownConfidence = sandbox.renderMessageEditors({messages: [{speaker: "other", text: "你好"}]}, "preview");
assert(unknownConfidence.includes("文字 未知"));
assert(!unknownConfidence.includes("100%"));
const confirmedMessage = sandbox.renderMessageEditors({messages: [{speaker: "me", text: "你好", speaker_source: "user_confirmed"}]}, "result");
assert(confirmedMessage.includes("已人工确认"));
assert(/backgroundInput.addEventListener\("input",[\s\S]*?invalidateTopic\(\)/.test(script));
assert(script.includes('event.target.matches("[data-field=\'text\']")'));
assert(script.includes('if (contextNeedsRegeneration)'));
assert(script.includes('getBackgroundContext() !== (data.background_context || "").trim()'));
console.log("Frontend topic source, escaping, edit invalidation and script syntax checks passed.");
