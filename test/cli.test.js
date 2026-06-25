const assert = require("assert");
const path = require("path");

const {
  getMimeType,
  isQuotaError,
  isModelUnavailableError,
  normalizeStats,
  normalizeChatCommand,
  normalizeConfig,
  parseInputArgs,
  parseToggleArg,
  pickInitialModel,
  pickNextModel,
  sanitizeFilePath
} = require("../cli");

(async () => {
  assert.strictEqual(getMimeType("screen.PNG"), "image/png");
  assert.strictEqual(getMimeType("notes.md"), "text/markdown");
  assert.strictEqual(getMimeType("unknown.ext"), "text/plain");
  assert.strictEqual(parseToggleArg("on", false), true);
  assert.strictEqual(parseToggleArg("off", true), false);
  assert.strictEqual(parseToggleArg(undefined, true), false);
  assert.strictEqual(isQuotaError(new Error("[429 Too Many Requests] quota exceeded")), true);
  assert.strictEqual(isQuotaError(new Error("fetch failed")), false);
  assert.strictEqual(isModelUnavailableError(new Error("[403 Forbidden] Your project has been denied access")), true);
  assert.strictEqual(pickInitialModel("flash", { fallbackModels: ["flash", "lite"] }, { requests: { flash: 20 } }), "lite");
  assert.strictEqual(pickNextModel("flash", { fallbackModels: ["flash", "lite", "gemma"] }, { requests: { lite: 500 } }, []), "gemma");
  assert.strictEqual(pickNextModel("flash", { fallbackModels: ["lite"] }, { requests: { lite: 500 } }, []), null);
  assert.strictEqual(normalizeConfig({ apiKey: "old-key" }).providers.google.apiKey, "old-key");
  assert.strictEqual(normalizeConfig({
    availableModels: {
      lite: { id: "gemini-3.1-flash-lite", supportsSearch: true, dailyLimit: 0 }
    }
  }).availableModels.lite.supportsSearch, false);
  assert.strictEqual(normalizeConfig({
    availableModels: {
      lite: { id: "gemini-3.1-flash-lite", supportsSearch: true, dailyLimit: 0 }
    }
  }).availableModels.lite.dailyLimit, 500);
  assert.strictEqual(normalizeChatCommand("gemini --sync-models"), "/sync-models");
  assert.strictEqual(normalizeChatCommand("gemini --models --refresh"), "/models --refresh");
  assert.strictEqual(normalizeChatCommand("gemini -m lite"), "/model lite");
  assert.strictEqual(normalizeChatCommand("hello"), "hello");

  assert.deepStrictEqual(normalizeStats({ counts: { flash: 1 }, lastReset: "old" }), {
    lastReset: "old",
    tokens: {},
    requests: {}
  });

  const parsed = await parseInputArgs(["Проверь код", "README.md", "*.js"]);
  assert.strictEqual(parsed.prompt, "Проверь код");
  assert(parsed.filesToProcess.includes("README.md"));
  assert(parsed.filesToProcess.some((file) => path.basename(file) === "cli.js"));

  assert.strictEqual(sanitizeFilePath("generated/result.txt"), path.resolve("generated/result.txt"));
  assert.strictEqual(sanitizeFilePath("../outside.txt"), null);

  console.log("cli tests passed");
})().catch((error) => {
  console.error(error);
  process.exit(1);
});
