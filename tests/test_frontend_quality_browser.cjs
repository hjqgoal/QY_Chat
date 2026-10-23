// Optional browser regression: PLAYWRIGHT_MODULE and BROWSER_CHANNEL may select a local runtime.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {chromium} = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const html = fs.readFileSync(path.join(__dirname, "../chat_reply_agent/static/index.html"), "utf8");
const originalTexts = ["行，那我先去了", "好啦，你们玩开心", "嗯，下次再说"];
const initialQuality = () => ({status: "pending", summary: "回复已生成，后台正在评估。", passed: false,
  diversity_status: "not_reviewed", items: originalTexts.map((_, candidate_index) => ({candidate_index, status: "pending", verdict: "unverified", assessment_source: "local"}))});

(async () => {
  const browser = await chromium.launch({headless: true, channel: process.env.BROWSER_CHANNEL || "chrome"});
  try {
    for (const width of [1280, 390]) {
      const page = await browser.newPage({viewport: {width, height: 900}});
      const errors = [];
      page.on("pageerror", error => errors.push(error.message));
      let phase = "running", jobId = "job1", texts = originalTexts.slice(), manualRequests = 0;
      let mode = "normal", stage = "review", highRisk = false;
      const reportFor = candidates => ({status: "reviewed", passed: true, diversity_status: "distinct", group_issues: [],
        review_mode: "quick", recommended_candidate_indexes: [0], recommendation_reason: "保留原有条件，接话自然。",
        group_reason: "分别表达接受、祝福和收尾。", summary: "已评估 3/3 条：3 条基本可用，0 条建议调整，0 条需补充关键信息。", items: candidates.map((text, candidate_index) => ({candidate_index, status: "reviewed",
          assessment_source: "read_only", verdict: "usable", evaluated_text: text, passed: true, naturalness: 4,
          relevance: 4, relation_fit: 4, scenario_fit: 4, style_fit: null,
          conclusion: "能接住这句话，没有新增承诺。", advisories: ["没有本人风格样本，风格项暂不计分。"],
          evidence: {naturalness: "<img src=x onerror=alert(1)>原文短句", style_fit: "本人样本不足"}}))});
      await page.route("**/*", async route => {
        const url = new URL(route.request().url());
        const method = route.request().method();
        const json = body => route.fulfill({json: body});
        if (url.pathname === "/") return route.fulfill({contentType: "text/html", body: html});
        if (url.pathname === "/api/status") return json({llm_configured: true, text_model: "test", vision_model: "test",
          memory_retrieval: {readiness: "not_verified", fallback_policy: "error", notice: "向量接口已配置，尚未验证连通性"}});
        if (url.pathname === "/api/contacts") return json({contacts: []});
        if (url.pathname === "/api/reply/from-chat") {
          const input = route.request().postDataJSON();
          return json({...input, extracted_chat: {messages: [], topic_source: "context", topic: "已有安排，临时补位提议被婉拒"},
            retrieval: {method: "semantic", candidate_count: 2, retrieved_count: 0,
              message: "本次已完成语义向量检索。没有达到相关性与时间衰减门槛的历史片段。<img src=x onerror=alert(1)>"},
            analysis: {tone: "简短", emotion: "可能失落，不能确定", attitude: "收回邀约", scene: "邀约收尾", risk_level: "low"},
            scenario: {label: "邀约", reply_goal: "自然收尾", confidence: .7},
            timing: {should_reply_now: true, suggested_delay: "方便时回复", reason: "无需刻意等待"}, strategy: "回应眼前这句话，不追加承诺。",
            replies: texts.map(text => ({text, style: "自然收尾", reason: "不额外承诺", source: "model"})),
            quality: initialQuality(), quality_job_id: jobId, quality_status: "pending", memory_status: "disabled", memory_persisted: false,
            quality_gate_required: highRisk,
            diagnostics: {total_ms: 15000, request_id: "offline-browser-test", stages: [{stage: "quality.enqueue", duration_ms: 1, status: "ok"}]}});
        }
        if (url.pathname === "/api/quality-jobs" && method === "POST") {
          manualRequests++;
          const input = route.request().postDataJSON();
          assert.equal(input.save_to_memory, false);
          texts = input.replies.map(r => r.text);
          jobId = `job${manualRequests + 1}`;
          phase = "running";
          return json({job_id: jobId, status: "pending", candidate_texts: texts, quality: initialQuality()});
        }
        if (url.pathname.startsWith("/api/quality-jobs/")) {
          if (method === "DELETE") return json({status: "cancelled"});
          const revised = ["好，那我先忙啦", "行，回头再说", texts[2]];
          if (mode === "repair") return json({job_id: jobId, status: phase, phase: stage, candidate_texts: texts,
            repair_status: phase === "completed" ? "completed" : "running",
            message: phase === "completed" ? "已修订并复评，改进版单独展示" : stage === "review" ? "后台评估中" : stage === "repair" ? "正在定向修订" : "正在复评改进版",
            quality: phase === "completed" ? {...reportFor(texts), passed: false} : null,
            revision_replies: phase === "completed" ? revised.map(text => ({text, source: "model", style: "自然", reason: "根据问题修订"})) : [],
            revision_quality: phase === "completed" ? reportFor(revised) : null,
            diagnostics: {total_ms: 33000, queue_wait_ms: 300, phase: stage, model_timeout_seconds: 90, total_budget_seconds: 240}});
          if (mode === "timeout" && phase === "timed_out") return json({job_id: jobId, status: phase, phase: "review", candidate_texts: texts,
            message: "评估超时，本任务已结束，不会自动重试。", quality: null,
            diagnostics: {total_ms: 90000, queue_wait_ms: 3500, phase: "review", error_code: "request_timeout", error_message: "模型接口请求超时。", worker_active: false}});
          if (mode === "partial" && phase === "completed") {
            const report = reportFor(texts);
            report.status = "partial";
            report.passed = false;
            report.summary = "部分评价未完成；已保留有效单条结果。";
            report.items[1] = {candidate_index: 1, status: "invalid_response", assessment_source: "local", verdict: "unverified", passed: false};
            return json({job_id: jobId, status: phase, candidate_texts: texts, quality: report,
              message: "已保留 2/3 条有效评价；本轮校验未全部完成，不自动修订或重试。",
              diagnostics: {total_ms: 15000, phase: "review", error_code: "incomplete_evidence",
                error_message: "部分评价缺少依据，已保留有效单条评价。",
                validation_errors: [{path: "items.1.review.evidence.relevance", type: "missing_or_blank_evidence"}]}});
          }
          return json({job_id: jobId, status: phase, candidate_texts: texts,
            diagnostics: {total_ms: 12500, queue_wait_ms: 200},
            quality: phase !== "completed" ? null : reportFor(texts)});
        }
        return route.fulfill({status: 404, body: "not found"});
      });
      await page.goto("http://qy.test/");
      await page.waitForFunction(() => document.querySelector('#retrieval-status')?.textContent.includes("严格模式"));
      assert((await page.locator('#retrieval-status').textContent()).includes("尚未验证连通性"));
      await page.locator('[name="background_context"]').fill("对方说算了算了你去玩，我想简短收尾。");
      await page.locator('[name="save_to_memory"]').uncheck();
      await page.locator('#reply-form button[type="submit"]').click();
      await page.waitForFunction(() => document.querySelector('[data-quality-job-state]')?.textContent.includes("后台评估中"));
      assert((await page.locator('[data-retrieval-status]').textContent()).includes("已完成语义向量检索"));
      assert.equal(await page.locator('[data-retrieval-status] img').count(), 0);
      assert.equal(await page.locator('[data-reply-index="0"] .quality-badge').textContent(), "后台评估中");
      const second = page.locator('[data-reply-index="1"] [data-reply-text]');
      await second.fill("我自己改的这句");
      await second.evaluate(input => {input.focus(); input.setSelectionRange(3, 3); input.dataset.identity = "keep";});
      phase = "completed";
      await page.waitForFunction(() => document.querySelector('[data-quality-job-state]')?.textContent === "后台评估已完成");
      assert.equal(await second.inputValue(), "我自己改的这句");
      assert.equal(await second.evaluate(input => input.dataset.identity), "keep");
      assert.equal(await second.evaluate(input => input.selectionStart), 3);
      assert.equal(await second.evaluate(input => document.activeElement === input), true);
      assert.equal(await page.locator('[data-reply-index="0"] .quality-badge').textContent(), "基本可用");
      assert.equal(await page.locator('[data-reply-index="1"] .quality-badge').textContent(), "已编辑，待确认");
      assert.equal(await page.locator('[data-quality-details] img').count(), 0);
      assert.equal(await page.locator('[data-quality-recommendation]').textContent(), "");
      assert((await page.locator('[data-quality-diagnostics]').textContent()).includes("12.50 秒"));
      assert((await page.locator('#result').textContent()).includes("15.00 秒（不含后台评估）"));
      await page.locator('[data-action="review-current"]').click();
      await page.waitForFunction(() => document.querySelector('[data-quality-job-state]')?.textContent.includes("后台评估中"));
      assert.equal(texts[1], "我自己改的这句");
      phase = "completed";
      await page.waitForFunction(() => [...document.querySelectorAll('[data-reply-index] .quality-badge')].every(e => e.textContent === "基本可用"));
      assert.equal(await second.inputValue(), "我自己改的这句");
      assert((await page.locator('[data-quality-recommendation]').textContent()).includes("候选 1"));
      assert((await page.locator('[data-reply-index="0"] [data-quality-details]').textContent()).includes("4/5"));
      assert((await page.locator('[data-reply-index="0"] [data-quality-details]').textContent()).includes("普通提醒"));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      if (process.env.SCREENSHOT_DIR) {
        await page.locator('[data-quality-total]').scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, `qy-background-quality-${width}.png`)});
        await page.locator('[data-reply-index="0"]').scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, `qy-quality-grades-${width}.png`)});
      }
      await page.locator('[data-action="review-current"]').click();
      await page.locator('[data-action="cancel-quality"]').click();
      phase = "completed";
      await page.waitForTimeout(2200);
      assert.equal(await page.locator('[data-quality-job-state]').textContent(), "评估已取消");
      assert.equal(await page.locator('[data-quality-recommendation]').textContent(), "");
      assert.equal(await second.inputValue(), "我自己改的这句");
      mode = "repair";
      highRisk = true;
      phase = "running";
      texts = originalTexts.slice();
      jobId = "gated-job";
      await page.locator('#reply-form button[type="submit"]').click();
      await page.waitForFunction(() => document.querySelector('[data-candidate-list]')?.hidden);
      assert.equal(await page.locator('[data-reply-index="0"] [data-reply-text]').isVisible(), false);
      stage = "repair";
      await page.waitForFunction(() => document.querySelector('[data-quality-job-state]')?.textContent.includes("定向修订"));
      stage = "recheck";
      await page.waitForFunction(() => document.querySelector('[data-quality-job-state]')?.textContent.includes("复评"));
      phase = "completed";
      await page.waitForFunction(() => document.querySelector('[data-revision-index="0"]'));
      assert.equal(await page.locator('[data-candidate-list]').isVisible(), true);
      assert.equal(await page.locator('[data-reply-index="0"] [data-reply-text]').inputValue(), originalTexts[0]);
      assert.equal(await page.locator('[data-quality-revisions] img').count(), 0);
      await page.locator('[data-action="use-revision"][data-index="0"]').click();
      assert.equal(await page.locator('[data-reply-index="0"] [data-reply-text]').inputValue(), "好，那我先忙啦");
      assert.equal(await page.locator('[data-quality-total]').textContent(), "待确认");
      assert.equal(await page.locator('[data-revision-index="0"] .quality-badge').textContent(), "已使用");
      assert.equal(await page.locator('[data-action="use-revision"][data-index="0"]').isDisabled(), true);
      await second.fill("改进版来了以后我又修改了");
      assert.equal(await page.locator('[data-action="use-revision"][data-index="1"]').isDisabled(), true);
      assert((await page.locator('[data-revision-index="1"] .quality-badge').getAttribute('class')).includes('failed'));
      if (process.env.SCREENSHOT_DIR) {
        await page.locator('[data-quality-revisions]').scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, `qy-quality-repair-${width}.png`)});
      }
      mode = "timeout";
      await page.locator('[data-action="review-current"]').click();
      phase = "timed_out";
      await page.waitForFunction(() => document.querySelector('[data-quality-job-state]')?.textContent.includes("不会自动重试"));
      assert((await page.locator('[data-quality-diagnostics]').textContent()).includes("模型接口请求超时"));
      assert((await page.locator('[data-quality-diagnostics]').textContent()).includes("90.00 秒"));
      assert.equal(await second.inputValue(), "改进版来了以后我又修改了");
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      mode = "partial";
      await page.locator('[data-action="review-current"]').click();
      phase = "completed";
      await page.waitForFunction(() => document.querySelector('[data-quality-job-state]')?.textContent.includes("2/3"));
      assert.equal(await page.locator('[data-reply-index="0"] .quality-badge').textContent(), "基本可用");
      assert.equal(await page.locator('[data-reply-index="1"] .quality-badge').textContent(), "复核未完成");
      assert.equal(await page.locator('[data-reply-index="2"] .quality-badge').textContent(), "基本可用");
      assert.equal(await page.locator('[data-quality-total]').textContent(), "待确认");
      assert.equal(await second.inputValue(), "改进版来了以后我又修改了");
      await page.locator('[data-quality-diagnostics] summary').click();
      assert((await page.locator('[data-quality-diagnostics]').textContent()).includes("本维度缺少评价依据"));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      if (process.env.SCREENSHOT_DIR) {
        await page.locator('[data-quality-total]').scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, `qy-partial-quality-${width}.png`)});
      }
      await page.evaluate(() => renderExtractionReview({
        messages: [
          {speaker: "me", side: "right", text: "周六见", text_confidence: .98, speaker_confidence: .98,
            needs_review: true, warnings: ["自动重新识别所得，请人工确认。"]},
          {speaker: "other", side: "left", text: "到时见", text_confidence: .98, speaker_confidence: .98,
            needs_review: true, warnings: ["自动重新识别所得，请人工确认。"]},
        ],
        confidence: .98, last_message_from_other: "到时见", source_image_count: 1,
        review_required: true, review_message_indexes: [0, 1],
        notes: ["已自动重新识别，请核对是否漏行。<img src=x onerror=alert(1)>"],
      }));
      assert.equal(await page.locator('.message.needs-review').count(), 2);
      assert((await page.locator('.review-banner.warning').textContent()).includes("2 条"));
      assert((await page.locator('#result').textContent()).includes("已自动重新识别"));
      assert.equal(await page.locator('#result img').count(), 0);
      assert.equal(await page.locator('[data-action="confirm-extraction"]').isEnabled(), true);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      await page.evaluate(() => renderExtractionReview({
        messages: [
          {speaker: "me", side: "right", text: "其实差不多", text_confidence: .98, speaker_confidence: .98},
          {speaker: "other", side: "left", text: "其实差不多", text_confidence: .98, speaker_confidence: .98},
        ],
        confidence: .98, last_message_from_other: "其实差不多", source_image_count: 1,
        review_required: false, review_message_indexes: [], notes: [],
      }));
      assert.equal(await page.locator('[data-message-index="0"] [data-field="speaker"]').inputValue(), "me");
      assert.equal(await page.locator('[data-message-index="1"] [data-field="speaker"]').inputValue(), "other");
      assert((await page.locator('#result').textContent()).includes("模型估计，非准确率"));
      await page.locator('[data-message-index="0"] [data-field="speaker"]').selectOption("other");
      assert.equal(await page.locator('[data-message-index="0"] [data-field="side"]').inputValue(), "left");
      const firstMessage = page.locator('.message').first();
      assert.equal(await firstMessage.locator('[data-speaker-label]').textContent(), "对方");
      assert.equal(await firstMessage.locator('[data-side-label]').textContent(), "左侧");
      assert.equal(await firstMessage.locator('.message-confidence').textContent(), "已修改，待确认");
      assert((await firstMessage.getAttribute('class')).split(' ').includes('other'));
      await page.locator('[data-message-index="1"] [data-field="text"]').fill("校对后的最后一句");
      assert.equal(await page.locator('[data-last-other]').textContent(), "校对后的最后一句");
      await page.locator('[data-message-index="1"] [data-field="side"]').selectOption("unknown");
      assert.equal(await page.locator('[data-message-index="1"] [data-field="speaker"]').inputValue(), "unknown");
      assert.equal(await page.locator('[data-last-other]').textContent(), "其实差不多");
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      if (process.env.SCREENSHOT_DIR) {
        await page.locator('#result').scrollIntoViewIfNeeded();
        await page.screenshot({path: path.join(process.env.SCREENSHOT_DIR, `qy-extraction-review-${width}.png`)});
      }
      assert.deepEqual(errors, []);
      await page.close();
      console.log(`Browser ${width}px: edits, cancellation, high-risk gate, separate revisions, explicit adoption, timeout diagnostics, OCR correction and layout passed.`);
    }
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
