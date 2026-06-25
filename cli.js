#!/usr/bin/env node

const { program } = require("commander");
const { GoogleGenerativeAI, SchemaType } = require("@google/generative-ai");
const fs = require("fs").promises;
const path = require("path");
const os = require("os");
const { glob } = require("glob");
const chalk = require("chalk");
const ora = require("ora");
const boxenModule = require("boxen");
const boxen = boxenModule.default || boxenModule;
const gradient = require("gradient-string");
const { marked } = require("marked");
const { markedTerminal } = require("marked-terminal");
const inquirer = require("inquirer");

// Поддержка .env файла
try { require("dotenv").config(); } catch {}

marked.use(markedTerminal({
  code: chalk.green,
  firstHeading: chalk.magenta.bold,
  heading: chalk.cyan.bold,
  strong: chalk.yellow.bold,
}));

// --- КОНФИГУРАЦИЯ И ЛИМИТЫ ---
const MAX_FILE_SIZE = 20 * 1024 * 1024;
const APP_CONFIG_DIR = process.env.GEMINI_CLI_HOME || path.join(os.homedir(), ".gemini-cli");
const LEGACY_STATS_FILE = path.join(__dirname, 'gemini-usage.json');
const LEGACY_CONFIG_FILE = path.join(__dirname, 'gemini-config.json');
const STATS_FILE = path.join(APP_CONFIG_DIR, 'usage.json');
const CONFIG_FILE = path.join(APP_CONFIG_DIR, 'config.json');
const RETRY_CODES = [429, 500, 502, 503];
const RETRIES = 3;
const RETRY_BASE_DELAY_MS = 600;
const GOOGLE_KNOWN_MODEL_META = {
  "gemini-3.1-flash-lite": { alias: "lite", dailyLimit: 500, autoFallback: true, supportsSearch: false },
  "gemini-3.1-flash-lite-preview": { alias: "lite-preview", dailyLimit: 500, autoFallback: true, supportsSearch: false },
  "gemma-4-31b-it": { alias: "gemma", dailyLimit: 1500, autoFallback: false, supportsSearch: false },
  "gemma-4-26b-a4b-it": { alias: "gemma-26b", dailyLimit: 1500, autoFallback: false, supportsSearch: false },
  "gemini-3.5-flash": { alias: "flash-3-5", dailyLimit: 20, autoFallback: true, supportsSearch: false },
  "gemini-3-flash-preview": { alias: "flash", dailyLimit: 20, autoFallback: true, supportsSearch: false },
  "gemini-2.5-flash-lite": { alias: "flash-lite-2-5", dailyLimit: 20, autoFallback: true, supportsSearch: true },
  "gemini-2.5-flash": { alias: "flash-2-5", dailyLimit: 20, autoFallback: true, supportsSearch: true },
  "gemini-2.5-flash-preview-tts": { dailyLimit: 10, autoFallback: false, supportsSearch: false, chat: false },
  "gemini-3.1-flash-tts-preview": { dailyLimit: 10, autoFallback: false, supportsSearch: false, chat: false },
  "gemini-2.0-flash": { dailyLimit: 0, autoFallback: false, chat: false },
  "gemini-2.0-flash-001": { dailyLimit: 0, autoFallback: false, chat: false },
  "gemini-2.0-flash-lite": { dailyLimit: 0, autoFallback: false, chat: false },
  "gemini-2.0-flash-lite-001": { dailyLimit: 0, autoFallback: false, chat: false },
  "gemini-2.5-pro": { dailyLimit: 0, autoFallback: false, chat: false },
  "gemini-3.1-pro-preview": { dailyLimit: 0, autoFallback: false, chat: false },
  "gemini-3.1-pro-preview-customtools": { dailyLimit: 0, autoFallback: false, chat: false },
  "deep-research-pro-preview-12-2025": { dailyLimit: 0, autoFallback: false, chat: false },
  "deep-research-preview-04-2026": { dailyLimit: 0, autoFallback: false, chat: false },
  "deep-research-max-preview-04-2026": { dailyLimit: 0, autoFallback: false, chat: false }
};
const PROVIDER_PRESETS = {
  google: {
    label: "Google Gemini",
    type: "google",
    env: "GEMINI_API_KEY",
    baseURL: "https://generativelanguage.googleapis.com",
    keyUrl: "https://aistudio.google.com/app/apikey",
    docsUrl: "https://ai.google.dev/gemini-api/docs",
    note: "Основной провайдер Gemini; ключ можно вставить в /settings или GEMINI_API_KEY."
  },
  deepseek: {
    label: "DeepSeek",
    type: "openai-compatible",
    env: "DEEPSEEK_API_KEY",
    baseURL: "https://api.deepseek.com",
    keyUrl: "https://platform.deepseek.com/api_keys",
    docsUrl: "https://api-docs.deepseek.com/",
    note: "OpenAI-compatible API. В новых docs основные модели: deepseek-v4-flash и deepseek-v4-pro."
  },
  qwen: {
    label: "Qwen / Alibaba Cloud Model Studio",
    type: "openai-compatible",
    env: "DASHSCOPE_API_KEY",
    baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    keyUrl: "https://help.aliyun.com/zh/model-studio/get-api-key",
    docsUrl: "https://help.aliyun.com/zh/model-studio/",
    note: "OpenAI-compatible endpoint DashScope. Ключ создаётся в Alibaba Cloud Model Studio / Bailian."
  },
  openaiCompatible: {
    label: "OpenAI-compatible",
    type: "openai-compatible",
    env: "OPENAI_COMPATIBLE_API_KEY",
    baseURL: "",
    keyUrl: "",
    docsUrl: "",
    note: "Для любого сервиса с /chat/completions: укажите API key, base URL и model id."
  }
};

let MODELS_CONFIG = {
  "flash": {
    provider: "google",
    id: "gemini-3-flash-preview",
    desc: "🚀 Основная модель. Скорость и новейшие знания 2026.",
    dailyLimit: 20
  },
  "lite": {
    provider: "google",
    id: "gemini-3.1-flash-lite",
    desc: "☁️ Высокие лимиты (500/день). Для рутины и тестов.",
    dailyLimit: 500
  },
  "flash-3-5": {
    provider: "google",
    id: "gemini-3.5-flash",
    desc: "⚡ Gemini 3.5 Flash. Запасной текстовый fallback.",
    dailyLimit: 20,
    supportsSearch: false,
    autoFallback: true
  },
  "flash-lite-2-5": {
    provider: "google",
    id: "gemini-2.5-flash-lite",
    desc: "⚡ Gemini 2.5 Flash Lite. Запасной текстовый fallback.",
    dailyLimit: 20,
    supportsSearch: true,
    autoFallback: true
  },
  "flash-2-5": {
    provider: "google",
    id: "gemini-2.5-flash",
    desc: "⚡ Gemini 2.5 Flash. Запасной текстовый fallback.",
    dailyLimit: 20,
    supportsSearch: true,
    autoFallback: true
  },
  "gemma": {
    provider: "google",
    id: "gemma-4-31b-it",
    desc: "📟 Огромные лимиты. Для массовой обработки текста.",
    dailyLimit: 1500,
    supportsSearch: false,
    autoFallback: false
  },
  "gemma-26b": {
    provider: "google",
    id: "gemma-4-26b-a4b-it",
    desc: "📟 Gemma 4 26B. Доступна вручную через /model.",
    dailyLimit: 1500,
    supportsSearch: false,
    autoFallback: false
  },
  "research": {
    provider: "google",
    id: "deep-research-pro-preview-12-2025",
    desc: "🔍 Глубокий поиск и создание документации.",
    dailyLimit: 500
  },
  "vision": {
    provider: "google",
    id: "gemini-3.1-flash-image-preview",
    desc: "👁️ Анализ скриншотов, макетов и графиков.",
    dailyLimit: 20
  },
  "deepseek": {
    provider: "deepseek",
    id: "deepseek-v4-flash",
    desc: "🌊 DeepSeek V4 Flash через OpenAI-compatible API.",
    dailyLimit: 0,
    supportsSearch: false,
    supportsStructuredOutput: true,
    autoFallback: true
  },
  "deepseek-pro": {
    provider: "deepseek",
    id: "deepseek-v4-pro",
    desc: "🌊 DeepSeek V4 Pro через OpenAI-compatible API.",
    dailyLimit: 0,
    supportsSearch: false,
    supportsStructuredOutput: true,
    autoFallback: true
  },
  "qwen": {
    provider: "qwen",
    id: "qwen-plus",
    desc: "☁️ Qwen Plus через Alibaba DashScope compatible API.",
    dailyLimit: 0,
    supportsSearch: false,
    supportsStructuredOutput: true,
    autoFallback: true
  },
  "qwen-max": {
    provider: "qwen",
    id: "qwen-max",
    desc: "☁️ Qwen Max через Alibaba DashScope compatible API.",
    dailyLimit: 0,
    supportsSearch: false,
    supportsStructuredOutput: true,
    autoFallback: true
  }
};
const BUILTIN_MODELS_CONFIG = JSON.parse(JSON.stringify(MODELS_CONFIG));

const systemInstructionBase = `Ты — экспертный CLI-помощник.`;
const DEFAULT_SETTINGS = {
  defaultModel: "lite",
  saveByDefault: false,
  dryRunByDefault: false,
  searchEnabled: true,
  maxOutputTokens: 8192,
  showWelcome: true,
  autoSyncModels: true,
  validateModelsOnSync: true,
  modelsSyncedAt: null,
  availableModels: null,
  providers: {
    google: {
      apiKey: null,
      baseURL: "https://generativelanguage.googleapis.com"
    },
    deepseek: {
      apiKey: null,
      baseURL: "https://api.deepseek.com"
    },
    qwen: {
      apiKey: null,
      baseURL: "https://dashscope.aliyuncs.com/compatible-mode/v1"
    },
    openaiCompatible: {
      apiKey: null,
      baseURL: "",
      model: ""
    }
  },
  autoFallback: true,
  fallbackModels: ["lite", "flash-3-5", "flash", "flash-lite-2-5", "flash-2-5", "deepseek", "qwen"]
};
const MODEL_SYNC_TTL_MS = 24 * 60 * 60 * 1000;

// --- ЛОГИКА ---

async function pathExists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

async function ensureConfigDir() {
  await fs.mkdir(APP_CONFIG_DIR, { recursive: true });
}

async function readJsonFile(filePath, fallback = {}) {
  try {
    return JSON.parse(await fs.readFile(filePath, 'utf8'));
  } catch {
    return fallback;
  }
}

async function readAppJson(primaryFile, legacyFile, fallback = {}) {
  if (await pathExists(primaryFile)) return readJsonFile(primaryFile, fallback);
  if (legacyFile && await pathExists(legacyFile)) return readJsonFile(legacyFile, fallback);
  return fallback;
}

async function writeAppJson(filePath, data) {
  await ensureConfigDir();
  await fs.writeFile(filePath, JSON.stringify(data, null, 2));
}

function normalizeModelCatalog(catalog = {}) {
  const normalized = {};
  for (const [alias, cfg] of Object.entries(catalog || {})) {
    const id = String(cfg.id || '').toLowerCase();
    const known = GOOGLE_KNOWN_MODEL_META[id];
    normalized[alias] = {
      ...cfg,
      provider: cfg.provider || "google",
      dailyLimit: known?.dailyLimit ?? cfg.dailyLimit ?? 0,
      supportsSearch: known?.supportsSearch ?? cfg.supportsSearch ?? inferGoogleSearchSupport(cfg.id),
      supportsStructuredOutput: cfg.supportsStructuredOutput ?? true,
      autoFallback: known?.autoFallback ?? cfg.autoFallback
    };
  }
  return normalized;
}

function inferGoogleSearchSupport(modelId = '') {
  const id = String(modelId).toLowerCase();
  const known = GOOGLE_KNOWN_MODEL_META[id];
  if (known?.supportsSearch !== undefined) return known.supportsSearch;
  if (id.includes('image') || id.includes('tts') || id.includes('lyria') || id.includes('robotics')) return false;
  if (id.startsWith('gemini-3')) return false;
  return id.includes('gemini');
}

function isAutoFallbackCandidate(modelKey) {
  const cfg = MODELS_CONFIG[modelKey];
  const id = String(cfg?.id || '').toLowerCase();
  if (!cfg) return false;
  if (getProviderType(cfg.provider) === 'openai-compatible') return cfg.autoFallback !== false;
  const known = GOOGLE_KNOWN_MODEL_META[id];
  if (known?.autoFallback !== undefined) return known.autoFallback && (cfg.dailyLimit || 0) > 0;
  if (id.includes('image') || id.includes('tts') || id.includes('lyria')) return false;
  if (id.includes('robotics') || id.includes('computer-use') || id.includes('research')) return false;
  return cfg.provider === 'google' && (id.includes('gemini') || id.includes('gemma'));
}

function shouldIncludeDiscoveredModel(modelId = '') {
  const id = String(modelId).toLowerCase();
  const known = GOOGLE_KNOWN_MODEL_META[id];
  if (known) return known.chat !== false && (known.dailyLimit || 0) > 0;
  if (id.includes('embedding') || id.includes('image') || id.includes('tts')) return false;
  if (id.includes('lyria') || id.includes('veo') || id.includes('imagen')) return false;
  if (id.includes('audio') || id.includes('live')) return false;
  if (id.includes('robotics') || id.includes('computer-use')) return false;
  if (id.includes('research') || id.includes('antigravity')) return false;
  return id.includes('gemini') || id.includes('gemma');
}

function normalizeConfig(rawConfig = {}) {
  const config = { ...DEFAULT_SETTINGS, ...rawConfig };
  config.providers = {};
  for (const [provider, defaults] of Object.entries(DEFAULT_SETTINGS.providers)) {
    config.providers[provider] = {
      ...defaults,
      ...(rawConfig.providers?.[provider] || {})
    };
  }
  for (const [provider, providerConfig] of Object.entries(rawConfig.providers || {})) {
    config.providers[provider] = {
      ...(PROVIDER_PRESETS[provider] ? { baseURL: PROVIDER_PRESETS[provider].baseURL } : {}),
      ...(config.providers[provider] || {}),
      ...providerConfig
    };
  }

  if (rawConfig.apiKey && !config.providers.google.apiKey) {
    config.providers.google.apiKey = rawConfig.apiKey;
  }
  delete config.apiKey;

  config.availableModels = config.availableModels ? normalizeModelCatalog(config.availableModels) : null;
  config.fallbackModels = Array.isArray(config.fallbackModels) && config.fallbackModels.length > 0
    ? config.fallbackModels
    : DEFAULT_SETTINGS.fallbackModels;
  return config;
}

function getProviderPreset(provider) {
  return PROVIDER_PRESETS[provider] || {
    label: provider,
    type: "openai-compatible",
    env: `${provider.toUpperCase()}_API_KEY`,
    baseURL: "",
    keyUrl: "",
    docsUrl: "",
    note: "Пользовательский OpenAI-compatible провайдер."
  };
}

function getProviderType(provider) {
  return getProviderPreset(provider).type;
}

function getProviderConfig(config, provider) {
  const preset = getProviderPreset(provider);
  const savedConfig = config.providers?.[provider] || {};
  const envKey = preset.env ? process.env[preset.env] : null;
  return {
    ...preset,
    ...savedConfig,
    apiKey: savedConfig.apiKey || envKey || null,
    provider
  };
}

function isProviderConfigured(config, provider) {
  const providerConfig = getProviderConfig(config, provider);
  return !!providerConfig.apiKey && (providerConfig.type === 'google' || !!providerConfig.baseURL);
}

async function readConfig() {
  const rawConfig = await readAppJson(CONFIG_FILE, LEGACY_CONFIG_FILE, {});
  const config = normalizeConfig(rawConfig);
  applyAvailableModels(config);
  return config;
}

async function writeConfig(config) {
  await writeAppJson(CONFIG_FILE, normalizeConfig(config));
}

function applyAvailableModels(config) {
  if (config?.availableModels && Object.keys(config.availableModels).length > 0) {
    MODELS_CONFIG = {
      ...normalizeModelCatalog(BUILTIN_MODELS_CONFIG),
      ...normalizeModelCatalog(config.availableModels)
    };
  } else {
    MODELS_CONFIG = normalizeModelCatalog(BUILTIN_MODELS_CONFIG);
  }
}

function getPreferredAlias(modelId, usedAliases) {
  const id = modelId.toLowerCase();
  const candidates = [];
  const knownAlias = GOOGLE_KNOWN_MODEL_META[id]?.alias;

  if (knownAlias) candidates.push(knownAlias);

  if (id.includes('flash-lite') || id.includes('lite')) candidates.push('lite');
  if (id.includes('flash') && !id.includes('image')) candidates.push('flash');
  if (id.includes('gemma')) candidates.push('gemma');
  if (id.includes('image') || id.includes('vision')) candidates.push('vision');
  if (id.includes('research')) candidates.push('research');

  const simple = modelId
    .replace(/^gemini-/, '')
    .replace(/^gemma-/, 'gemma-')
    .replace(/-preview.*$/, '')
    .replace(/-latest$/, '')
    .replace(/[^a-zA-Z0-9]+/g, '-')
    .replace(/^-|-$/g, '')
    .toLowerCase();
  candidates.push(simple || modelId.toLowerCase());

  for (const candidate of candidates) {
    if (candidate && !usedAliases.has(candidate)) return candidate;
  }

  let index = 2;
  while (usedAliases.has(`${candidates[0]}-${index}`)) index += 1;
  return `${candidates[0]}-${index}`;
}

function buildModelCatalog(apiModels) {
  const catalog = {};
  const usedAliases = new Set();

  for (const model of apiModels || []) {
    if (!model.supportedGenerationMethods?.includes('generateContent')) continue;
    const id = model.name?.replace(/^models\//, '');
    if (!id) continue;
    if (!shouldIncludeDiscoveredModel(id)) continue;

    const alias = getPreferredAlias(id, usedAliases);
    const known = GOOGLE_KNOWN_MODEL_META[id.toLowerCase()];
    usedAliases.add(alias);
    catalog[alias] = {
      provider: 'google',
      id,
      desc: model.displayName || model.description || 'Доступна для generateContent по вашему API ключу',
      dailyLimit: known?.dailyLimit ?? MODELS_CONFIG[alias]?.dailyLimit ?? 0,
      supportsSearch: known?.supportsSearch ?? inferGoogleSearchSupport(id),
      supportsStructuredOutput: true,
      autoFallback: known?.autoFallback ?? isAutoFallbackCandidate(alias),
      source: 'api'
    };
  }

  return catalog;
}

async function fetchAvailableModels(apiKey) {
  const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}&pageSize=1000`);
  if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
  const data = await res.json();
  return buildModelCatalog(data.models || []);
}

async function canUseModel(apiKey, modelId) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${apiKey}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ role: 'user', parts: [{ text: 'ping' }] }],
      generationConfig: { maxOutputTokens: 1 }
    })
  });

  if (res.ok) return { ok: true };
  return { ok: false, status: res.status, body: await res.text() };
}

async function validateModelCatalog(apiKey, catalog, spinner = null) {
  const entries = Object.entries(catalog);
  const validated = {};
  const denied = [];

  for (let index = 0; index < entries.length; index += 1) {
    const [alias, cfg] = entries[index];
    if (spinner) spinner.text = chalk.cyan(`Проверяю доступ к моделям ${index + 1}/${entries.length}: ${cfg.id}`);

    try {
      const result = await canUseModel(apiKey, cfg.id);
      if (result.ok) {
        validated[alias] = { ...cfg, validatedAt: new Date().toISOString() };
      } else {
        denied.push({ alias, id: cfg.id, reason: `HTTP ${result.status}` });
      }
    } catch (error) {
      denied.push({ alias, id: cfg.id, reason: error.message });
    }
  }

  return { validated, denied };
}

async function syncModels(apiKey, config, { force = false, silent = false } = {}) {
  const lastSync = config.modelsSyncedAt ? new Date(config.modelsSyncedAt).getTime() : 0;
  const hasFreshCache = config.availableModels && Object.keys(config.availableModels).length > 0 && Date.now() - lastSync < MODEL_SYNC_TTL_MS;

  if (!force && hasFreshCache) {
    applyAvailableModels(config);
    return { synced: false, fromCache: true, models: MODELS_CONFIG };
  }

  const spinner = silent ? null : ora(chalk.cyan('Синхронизирую доступные модели по API ключу...')).start();
  try {
    let catalog = await fetchAvailableModels(apiKey);
    if (Object.keys(catalog).length === 0) {
      throw new Error('API не вернул моделей с поддержкой generateContent');
    }

    let deniedModels = [];
    if (config.validateModelsOnSync) {
      const validation = await validateModelCatalog(apiKey, catalog, spinner);
      catalog = validation.validated;
      deniedModels = validation.denied;
      if (Object.keys(catalog).length === 0) {
        throw new Error('Ни одна модель не прошла проверку доступа generateContent');
      }
    }

    config.availableModels = catalog;
    config.deniedModels = deniedModels;
    config.modelsSyncedAt = new Date().toISOString();
    if (!catalog[config.defaultModel]) {
      const preferred = catalog.lite ? 'lite' : (catalog.gemma ? 'gemma' : (catalog.flash ? 'flash' : Object.keys(catalog)[0]));
      config.defaultModel = preferred;
    }
    await writeConfig(config);
    applyAvailableModels(config);
    if (spinner) {
      const deniedText = deniedModels.length > 0 ? chalk.gray(`, скрыто недоступных: ${deniedModels.length}`) : '';
      spinner.succeed(chalk.green(`Доступные модели обновлены: ${Object.keys(catalog).length}`) + deniedText);
    }
    return { synced: true, fromCache: false, models: catalog, deniedModels };
  } catch (error) {
    if (config.availableModels && Object.keys(config.availableModels).length > 0) {
      applyAvailableModels(config);
      if (spinner) spinner.warn(chalk.yellow(`Не удалось обновить модели, использую кеш: ${error.message}`));
      return { synced: false, fromCache: true, models: MODELS_CONFIG, error };
    }
    if (spinner) spinner.fail(chalk.red('Не удалось получить список доступных моделей.'));
    throw error;
  }
}

function getTokenPrice(modelKey) {
  if (modelKey === "gemma") return 0.03;
  if (modelKey === "lite") return 0.075;
  return 0.15;
}

function resolveModelKey(modelKey) {
  if (MODELS_CONFIG[modelKey]) return modelKey;
  const fallback = MODELS_CONFIG.lite ? "lite" : (MODELS_CONFIG.gemma ? "gemma" : (MODELS_CONFIG.flash ? "flash" : Object.keys(MODELS_CONFIG)[0]));
  console.warn(chalk.yellow(`⚠️  Модель "${modelKey}" не найдена, использую ${fallback}.`));
  return fallback;
}

function normalizeStats(stats = {}) {
  if (stats.counts && !stats.tokens) {
    return { lastReset: stats.lastReset, tokens: {}, requests: {} };
  }
  return {
    lastReset: stats.lastReset || new Date().toDateString(),
    tokens: stats.tokens || {},
    requests: stats.requests || {}
  };
}

async function processSavedFiles(filesArray, isDryRun) {
  if (!filesArray || filesArray.length === 0) {
    console.log(chalk.yellow(`ℹ️  Файлы для сохранения не предложены.`));
    return;
  }
  for (const file of filesArray) {
    const rawPath = file.path;
    const fileContent = file.content;
    const safePath = sanitizeFilePath(rawPath);
    if (!safePath) {
      console.warn(chalk.yellow(`⚠️  Пропущен небезопасный путь: ${rawPath}`));
      continue;
    }
    if (isDryRun) { console.log(chalk.cyan(`[dry-run] Сохранил бы: ${rawPath}`)); continue; }
    try {
      await fs.mkdir(path.dirname(safePath), { recursive: true });
      await fs.writeFile(safePath, fileContent, 'utf8');
      console.log(chalk.green(`💾 Сохранен: `) + chalk.bold(rawPath));
    } catch (err) {
      console.error(chalk.red(`❌ Ошибка сохранения ${rawPath}:`), err.message);
    }
  }
}

async function getDefaultModel() {
  const config = await readConfig();
  return resolveModelKey(config.defaultModel || "flash");
}

async function setDefaultModel(modelKey) {
  const config = await readConfig();
  config.defaultModel = modelKey;
  await writeConfig(config);
}

async function updateAndGetUsage(modelKey, tokenCount = 0) {
  let stats = normalizeStats(await readAppJson(STATS_FILE, LEGACY_STATS_FILE, {
    lastReset: new Date().toDateString(),
    tokens: {},
    requests: {}
  }));
  
  // Сброс статистики при смене дня
  if (stats.lastReset !== new Date().toDateString()) {
    stats.tokens = {};
    stats.requests = {};
    stats.lastReset = new Date().toDateString();
  }
  
  // Подсчет токенов и запросов
  stats.tokens[modelKey] = (stats.tokens[modelKey] || 0) + tokenCount;
  stats.requests[modelKey] = (stats.requests[modelKey] || 0) + 1;
  
  await writeAppJson(STATS_FILE, stats);
  return stats;
}

async function getUsageSnapshot() {
  const stats = normalizeStats(await readAppJson(STATS_FILE, LEGACY_STATS_FILE, {
    lastReset: new Date().toDateString(),
    tokens: {},
    requests: {}
  }));
  if (stats.lastReset !== new Date().toDateString()) {
    stats.tokens = {};
    stats.requests = {};
    stats.lastReset = new Date().toDateString();
  }
  return stats;
}

function showLimitWarning(modelKey, stats) {
  const cfg = MODELS_CONFIG[modelKey];
  const requests = stats.requests?.[modelKey] || 0;
  if (!cfg?.dailyLimit) return;
  const remaining = cfg.dailyLimit - requests;
  if (remaining <= 0) {
    console.warn(chalk.red(`⚠️  Локальный дневной лимит для "${modelKey}" уже достигнут (${requests}/${cfg.dailyLimit}). Запрос всё равно будет отправлен.`));
  } else if (remaining <= Math.max(3, Math.ceil(cfg.dailyLimit * 0.1))) {
    console.warn(chalk.yellow(`⚠️  До локального лимита "${modelKey}" осталось ${remaining} запрос(ов).`));
  }
}

function formatRemainingLimit(modelKey, requests) {
  const limit = MODELS_CONFIG[modelKey]?.dailyLimit;
  if (!limit) return 'нет данных';
  return `${Math.max(0, limit - requests)}/${limit}`;
}

function hasLocalLimitRemaining(modelKey, stats) {
  const limit = MODELS_CONFIG[modelKey]?.dailyLimit;
  if (!limit) return true;
  return (stats.requests?.[modelKey] || 0) < limit;
}

function getFallbackChain(config = {}) {
  const configured = Array.isArray(config.fallbackModels) ? config.fallbackModels : [];
  return [...new Set(configured)].filter(modelKey => {
    const cfg = MODELS_CONFIG[modelKey];
    return cfg && isAutoFallbackCandidate(modelKey) && isProviderConfigured(config, cfg.provider || 'google');
  });
}

function pickNextModel(currentModelKey, config = {}, stats = { requests: {} }, triedModels = []) {
  const tried = new Set([currentModelKey, ...triedModels]);
  const chain = getFallbackChain(config);
  return chain.find(modelKey => !tried.has(modelKey) && hasLocalLimitRemaining(modelKey, stats)) || null;
}

function pickInitialModel(modelKey, config = {}, stats = { requests: {} }) {
  if (hasLocalLimitRemaining(modelKey, stats)) return modelKey;
  return pickNextModel(modelKey, config, stats, []) || modelKey;
}

function showConfiguredModels() {
  console.log(chalk.magenta.bold('\n🤖 НАСТРОЕННЫЕ МОДЕЛИ\n'));
  for (const [key, cfg] of Object.entries(MODELS_CONFIG)) {
    console.log(`${chalk.yellow(key.padEnd(10))} ${chalk.cyan(cfg.id)}`);
    console.log(chalk.gray(`           ${cfg.desc}`));
    console.log(chalk.gray(`           Провайдер: ${cfg.provider || 'google'} · Локальный лимит: ${cfg.dailyLimit ? `${cfg.dailyLimit}/день` : 'не задан'}${cfg.source === 'api' ? ' · synced' : ''}\n`));
  }
}

function showDeniedModels(config) {
  if (!config?.deniedModels || config.deniedModels.length === 0) return;
  console.log(chalk.yellow.bold('\nСКРЫТЫ НЕДОСТУПНЫЕ МОДЕЛИ\n'));
  for (const model of config.deniedModels.slice(0, 20)) {
    console.log(chalk.gray(`${model.alias.padEnd(12)} ${model.id} · ${model.reason}`));
  }
  if (config.deniedModels.length > 20) {
    console.log(chalk.gray(`...и ещё ${config.deniedModels.length - 20}`));
  }
  console.log();
}

async function resetStats() {
  const stats = {
    lastReset: new Date().toDateString(),
    tokens: {},
    requests: {}
  };
  await writeAppJson(STATS_FILE, stats);
  console.log(chalk.green(`✅ Статистика сброшена: ${STATS_FILE}`));
}

function showConfigPath() {
  console.log(chalk.magenta.bold('\n⚙️  ПУТИ GEMINI CLI\n'));
  console.log(`${chalk.bold('Конфиг:')}     ${CONFIG_FILE}`);
  console.log(`${chalk.bold('Статистика:')} ${STATS_FILE}`);
  console.log(`${chalk.bold('Папка:')}      ${APP_CONFIG_DIR}\n`);
}

function getLogo() {
  return gradient.pastel.multiline([
    "   ____ _____ __  __ ___ _   _ ___",
    "  / ___| ____|  \\/  |_ _| \\ | |_ _|",
    " | |  _|  _| | |\\/| || ||  \\| || |",
    " | |_| | |___| |  | || || |\\  || |",
    "  \\____|_____|_|  |_|___|_| \\_|___|"
  ].join("\n"));
}

function getModeLabel(saveFiles, dryRun) {
  if (dryRun) return chalk.yellow("dry-run");
  if (saveFiles) return chalk.green("save");
  return chalk.cyan("chat");
}

function getProviderStatusLine(config, provider) {
  const preset = getProviderPreset(provider);
  const connected = isProviderConfigured(config, provider);
  return `${connected ? chalk.green('on ') : chalk.gray('off')} ${preset.label}`;
}

function getCompactFallbackChain(config) {
  const chain = getFallbackChain(config);
  if (chain.length === 0) return chalk.gray('off');
  return chain.slice(0, 5).map(modelKey => chalk.yellow(modelKey)).join(chalk.gray(' -> ')) +
    (chain.length > 5 ? chalk.gray(` -> +${chain.length - 5}`) : '');
}

function parseToggleArg(value, currentValue) {
  if (!value) return !currentValue;
  const normalized = value.toLowerCase();
  if (['on', 'yes', 'true', '1', 'вкл', 'да'].includes(normalized)) return true;
  if (['off', 'no', 'false', '0', 'выкл', 'нет'].includes(normalized)) return false;
  return !currentValue;
}

function normalizeChatCommand(input) {
  const trimmed = input.trim();
  const lower = trimmed.toLowerCase();
  if (!lower.startsWith('gemini')) return trimmed;

  const parts = trimmed.split(/\s+/);
  if (parts[0].toLowerCase() !== 'gemini') return trimmed;
  const args = parts.slice(1);
  if (args.length === 0) return '/home';

  const first = args[0];
  const rest = args.slice(1).join(' ');
  const map = {
    '--help': '/help',
    '-h': '/help',
    '--sync-models': '/sync-models',
    '--models': '/models',
    '--scan': '/sync-models',
    '--doctor': '/doctor',
    '--providers': '/providers',
    '--settings': '/settings',
    '-c': '/settings',
    '--config': '/settings',
    '--stats': '/stats',
    '--reset-stats': '/reset-stats',
    '--config-path': '/path'
  };

  if (first === '-m' || first === '--model') {
    return `/model ${rest}`.trim();
  }
  if (first === '-i' || first === '--interactive') {
    return '/home';
  }
  if (first === '--save') {
    return '/save on';
  }
  if (first === '-d' || first === '--dry-run') {
    return '/dry on';
  }
  if (map[first]) {
    return `${map[first]} ${rest}`.trim();
  }

  return trimmed;
}

async function showHomeScreen(modelKey, config, session) {
  if (!config.showWelcome) return;
  modelKey = resolveModelKey(modelKey);

  const stats = await getUsageSnapshot();
  const requests = stats.requests[modelKey] || 0;
  const tokens = stats.tokens[modelKey] || 0;
  const cfg = MODELS_CONFIG[modelKey];
  const provider = cfg.provider || 'google';
  const providerLabel = getProviderPreset(provider).label;
  const remaining = formatRemainingLimit(modelKey, requests);
  const fallbackLine = config.autoFallback ? getCompactFallbackChain(config) : chalk.gray('off');
  const providersLine = ['google', 'deepseek', 'qwen']
    .map(providerKey => getProviderStatusLine(config, providerKey))
    .join(chalk.gray('  |  '));
  const modeBits = [
    getModeLabel(session.saveFiles, session.dryRun),
    session.searchEnabled && cfg.supportsSearch !== false ? chalk.cyan('search') : chalk.gray('no-search'),
    config.autoFallback ? chalk.green('auto-fallback') : chalk.gray('manual-fallback')
  ].join(chalk.gray(' / '));
  const body = [
    `${chalk.magenta.bold('SESSION')}`,
    `  ${chalk.bold('Model')}     ${chalk.yellow(modelKey)} ${chalk.gray(`(${cfg.id})`)}`,
    `  ${chalk.bold('Provider')}  ${providerLabel}`,
    `  ${chalk.bold('Mode')}      ${modeBits}`,
    `  ${chalk.bold('Today')}     ${requests} req · ${tokens} tokens · left ${remaining}`,
    "",
    `${chalk.magenta.bold('PROVIDERS')}`,
    `  ${providersLine}`,
    `  ${chalk.bold('Fallback')}  ${fallbackLine}`,
    "",
    `${chalk.magenta.bold('ACTIONS')}`,
    `  ${chalk.cyan('/model')} choose model        ${chalk.cyan('/providers')} keys + links`,
    `  ${chalk.cyan('/settings')} configure        ${chalk.cyan('/stats')} usage today`,
    `  ${chalk.cyan('/files <glob>')} attach files  ${chalk.cyan('/help')} all commands`,
    "",
    `${chalk.gray('Config:')} ${CONFIG_FILE}`
  ].join("\n");

  console.log(`\n${getLogo()}\n`);
  console.log(boxen(body, {
    padding: 1,
    borderColor: 'magenta',
    borderStyle: 'round',
    title: 'GEMINI CLI',
    titleAlignment: 'center'
  }));
}

function showChatHelp() {
  console.log(boxen([
    `${chalk.magenta.bold('Chat')}`,
    `  ${chalk.cyan("/model, /m [key]")}    выбрать модель для этой сессии`,
    `  ${chalk.cyan('/files <glob>')}        добавить файлы к следующему сообщению`,
    `  ${chalk.cyan("/clear")}              начать новый чат без истории`,
    `  ${chalk.cyan("/home")}               показать стартовый экран`,
    "",
    `${chalk.magenta.bold('Settings')}`,
    `  ${chalk.cyan("/providers")}          ключи, ссылки и подключённые провайдеры`,
    `  ${chalk.cyan("/settings, /config")}  открыть настройки CLI`,
    `  ${chalk.cyan("/save [on|off]")}      режим сохранения файлов`,
    `  ${chalk.cyan("/dry [on|off]")}       dry-run без записи файлов`,
    `  ${chalk.cyan("/search [on|off]")}    Google Search, если модель поддерживает`,
    "",
    `${chalk.magenta.bold('Project')}`,
    `  ${chalk.cyan("/models")}             показать локально настроенные модели`,
    `  ${chalk.cyan("/sync-models")}        обновить список моделей Google`,
    `  ${chalk.cyan("/stats [key]")}        показать статистику`,
    `  ${chalk.cyan("/doctor")}             проверить окружение`,
    `  ${chalk.cyan("/path")}               пути к конфигу и статистике`,
    `  ${chalk.cyan("/reset-stats")}        сбросить локальную статистику`,
    "",
    `  ${chalk.cyan("/exit, /q")}           выйти`
  ].join("\n"), { padding: 1, borderColor: 'cyan', borderStyle: 'round', title: 'COMMANDS' }));
}

function showProvidersGuide(config) {
  const lines = [];
  for (const [provider, preset] of Object.entries(PROVIDER_PRESETS)) {
    const providerConfig = getProviderConfig(config, provider);
    const configured = isProviderConfigured(config, provider);
    lines.push(`${configured ? chalk.green('✓') : chalk.yellow('!')} ${chalk.bold(preset.label)} ${chalk.gray(`(${provider})`)}`);
    lines.push(`  ${chalk.gray('Key:')} ${providerConfig.apiKey ? 'сохранен' : `не задан${preset.env ? ` · env ${preset.env}` : ''}`}`);
    if (providerConfig.baseURL) lines.push(`  ${chalk.gray('Base URL:')} ${providerConfig.baseURL}`);
    if (preset.keyUrl) lines.push(`  ${chalk.gray('Где взять ключ:')} ${preset.keyUrl}`);
    if (preset.docsUrl) lines.push(`  ${chalk.gray('Docs:')} ${preset.docsUrl}`);
    lines.push(`  ${chalk.gray(preset.note)}`);
    lines.push('');
  }
  lines.push(`${chalk.cyan('/settings')} → ${chalk.gray('API ключи провайдеров')} — вставить ключ или настроить OpenAI-compatible.`);
  console.log(boxen(lines.join('\n'), { padding: 1, borderColor: 'cyan', borderStyle: 'round', title: 'PROVIDERS' }));
}

async function promptModelChoice(currentModelKey) {
  const { modelKey } = await inquirer.prompt([{
    type: 'list',
    name: 'modelKey',
    message: 'Выберите модель:',
    default: currentModelKey,
    choices: Object.keys(MODELS_CONFIG).map(k => ({
      name: `${k.padEnd(8)} | ${MODELS_CONFIG[k].id} | ${MODELS_CONFIG[k].desc}`,
      value: k
    }))
  }]);
  return modelKey;
}

async function promptFallbackModel(currentModelKey) {
  const choices = Object.keys(MODELS_CONFIG)
    .filter(key => key !== currentModelKey)
    .map(key => ({
      name: `${key.padEnd(12)} | ${MODELS_CONFIG[key].id}`,
      value: key
    }));

  if (choices.length === 0) return null;
  choices.push({ name: 'Остаться на текущей модели', value: null });

  const { modelKey } = await inquirer.prompt([{
    type: 'list',
    name: 'modelKey',
    message: `Похоже, у "${currentModelKey}" закончилась квота. Переключиться?`,
    choices
  }]);

  return modelKey;
}

async function offerQuotaModelSwitch(currentModelKey, setModel) {
  const nextModelKey = await promptFallbackModel(currentModelKey);
  if (!nextModelKey) return currentModelKey;
  await setModel(nextModelKey);
  console.log(chalk.green(`✓ Переключился на ${nextModelKey}. История чата очищена.`));
  return nextModelKey;
}

async function editProviderSettings(config) {
  const providerChoices = Object.entries(PROVIDER_PRESETS).map(([provider, preset]) => {
    const providerConfig = getProviderConfig(config, provider);
    const status = providerConfig.apiKey ? 'ключ сохранен' : 'ключ не задан';
    return {
      name: `${preset.label.padEnd(34)} | ${status}`,
      value: provider
    };
  });
  providerChoices.push({ name: 'Назад', value: null });

  const { provider } = await inquirer.prompt([{
    type: 'list',
    name: 'provider',
    message: 'Провайдер',
    choices: providerChoices
  }]);
  if (!provider) return config;

  config.providers[provider] = config.providers[provider] || {};
  const preset = getProviderPreset(provider);
  const providerConfig = getProviderConfig(config, provider);
  console.log();
  showProvidersGuide(config);

  const choices = [
    { name: 'Сохранить/заменить API key', value: 'setKey' },
    { name: 'Удалить API key из config.json', value: 'clearKey' }
  ];
  if (preset.type === 'openai-compatible') {
    choices.push({ name: 'Изменить base URL', value: 'baseURL' });
    if (provider === 'openaiCompatible') choices.push({ name: 'Изменить model id', value: 'model' });
  }
  choices.push({ name: 'Назад', value: 'back' });

  const { action } = await inquirer.prompt([{
    type: 'list',
    name: 'action',
    message: `${preset.label}: ${providerConfig.apiKey ? 'ключ сохранен' : 'ключ не задан'}`,
    choices
  }]);

  if (action === 'setKey') {
    const { apiKey } = await inquirer.prompt([{
      type: 'password',
      name: 'apiKey',
      message: `Введите API key для ${preset.label}:`,
      mask: '*',
      validate: value => value && value.trim().length > 0 ? true : 'Ключ не может быть пустым'
    }]);
    config.providers[provider].apiKey = apiKey.trim();
  }
  if (action === 'clearKey') {
    config.providers[provider].apiKey = null;
  }
  if (action === 'baseURL') {
    const { baseURL } = await inquirer.prompt([{
      type: 'input',
      name: 'baseURL',
      message: 'Base URL:',
      default: providerConfig.baseURL || preset.baseURL,
      validate: value => value && value.trim().length > 0 ? true : 'Base URL не может быть пустым'
    }]);
    config.providers[provider].baseURL = baseURL.trim().replace(/\/+$/, '');
  }
  if (action === 'model') {
    const { model } = await inquirer.prompt([{
      type: 'input',
      name: 'model',
      message: 'Model id:',
      default: providerConfig.model || 'gpt-4o-mini',
      validate: value => value && value.trim().length > 0 ? true : 'Model id не может быть пустым'
    }]);
    config.providers[provider].model = model.trim();
    config.availableModels = config.availableModels || {};
    config.availableModels.openai = {
      provider,
      id: model.trim(),
      desc: 'Пользовательская OpenAI-compatible модель',
      dailyLimit: 0,
      supportsSearch: false,
      supportsStructuredOutput: true,
      autoFallback: true
    };
  }

  return config;
}

async function openSettingsMenu(config) {
  let nextConfig = normalizeConfig(config);

  while (true) {
    const { action } = await inquirer.prompt([{
      type: 'list',
      name: 'action',
        message: 'Настройки Gemini CLI',
        choices: [
        { name: 'API ключи провайдеров', value: 'providers' },
        { name: `Модель по умолчанию: ${nextConfig.defaultModel}`, value: 'model' },
        { name: `Автозамена модели: ${nextConfig.autoFallback ? 'включена' : 'выключена'}`, value: 'fallback' },
        { name: `Сохранять файлы по умолчанию: ${nextConfig.saveByDefault ? 'да' : 'нет'}`, value: 'save' },
        { name: `Dry-run по умолчанию: ${nextConfig.dryRunByDefault ? 'да' : 'нет'}`, value: 'dry' },
        { name: `Google Search: ${nextConfig.searchEnabled ? 'включен' : 'выключен'}`, value: 'search' },
        { name: `Автообновление моделей: ${nextConfig.autoSyncModels ? 'включено' : 'выключено'}`, value: 'autosync' },
        { name: `Проверять доступ к моделям: ${nextConfig.validateModelsOnSync ? 'да' : 'нет'}`, value: 'validateModels' },
        { name: `Max output tokens: ${nextConfig.maxOutputTokens}`, value: 'tokens' },
        { name: `Показывать стартовый экран: ${nextConfig.showWelcome ? 'да' : 'нет'}`, value: 'welcome' },
        { name: 'Показать путь к конфигу', value: 'path' },
        { name: 'Готово', value: 'done' }
      ]
    }]);

    if (action === 'done') break;
    if (action === 'path') {
      showConfigPath();
      continue;
    }
    if (action === 'providers') {
      nextConfig = await editProviderSettings(nextConfig);
    }
    if (action === 'model') nextConfig.defaultModel = await promptModelChoice(nextConfig.defaultModel);
    if (action === 'fallback') nextConfig.autoFallback = !nextConfig.autoFallback;
    if (action === 'save') nextConfig.saveByDefault = !nextConfig.saveByDefault;
    if (action === 'dry') nextConfig.dryRunByDefault = !nextConfig.dryRunByDefault;
    if (action === 'search') nextConfig.searchEnabled = !nextConfig.searchEnabled;
    if (action === 'autosync') nextConfig.autoSyncModels = !nextConfig.autoSyncModels;
    if (action === 'validateModels') nextConfig.validateModelsOnSync = !nextConfig.validateModelsOnSync;
    if (action === 'welcome') nextConfig.showWelcome = !nextConfig.showWelcome;
    if (action === 'tokens') {
      const { maxOutputTokens } = await inquirer.prompt([{
        type: 'number',
        name: 'maxOutputTokens',
        message: 'Max output tokens:',
        default: nextConfig.maxOutputTokens,
        validate: value => Number.isInteger(value) && value > 0 ? true : 'Введите положительное число'
      }]);
      nextConfig.maxOutputTokens = maxOutputTokens;
    }
    await writeConfig(nextConfig);
    console.log(chalk.green('✓ Настройки сохранены'));
  }

  return nextConfig;
}

async function scanModels() {
  const spinner = ora(chalk.cyan('Запрашиваю доступные модели у Google...')).start();
  try {
    const key = await getApiKey();
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${key}&pageSize=100`);
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const data = await res.json();
    spinner.succeed(chalk.green('Список моделей получен:\n'));

    console.log(chalk.bold.underline('Актуальные ID для вашего ключа:'));

    for (const m of data.models || []) {
      if (m.supportedGenerationMethods?.includes('generateContent')) {
        const modelId = m.name.replace('models/', '');
        const limits = m.limits || {};
        const rpm = limits.requestsPerMinute ?? '?';
        const tpm = limits.tokensPerMinute ?? '?';
        const info = `[RPM: ${rpm}, TPM: ${tpm}]`;
        console.log(`${chalk.yellow('→')} ${chalk.cyan(modelId.padEnd(35))} ${chalk.gray(info)}`);
      }
    }

    console.log(chalk.gray('\nЧтобы сохранить этот список для выбора в CLI, выполните: gemini --sync-models'));
  } catch (error) {
    spinner.fail(chalk.red('Не удалось получить список моделей.'));
    console.error(chalk.red("❌ Детали:"), error.message);
    console.log(chalk.gray("\nПопробуйте обновить библиотеку: npm install @google/generative-ai@latest"));
  }
  setTimeout(() => process.exit(0), 100);
}

async function syncModelsCommand(config) {
  try {
    const key = await getApiKey();
    await syncModels(key, config, { force: true });
    config = await readConfig();
    showConfiguredModels();
    showDeniedModels(config);
  } catch (error) {
    console.error(chalk.red("❌ Не удалось синхронизировать модели:"), error.message);
    console.log(chalk.gray("Проверьте API ключ, интернет и доступность Google Generative Language API."));
  }
}

async function getApiKey(provider = 'google') {
  const key = await findApiKey(provider);
  if (key) return key;

  console.error(chalk.red("\n❌ Ошибка: API ключ не найден."));
  console.log(chalk.gray(`Пожалуйста, установите ключ с помощью команды: ${chalk.cyan('gemini --set-key "ваш_ключ"')}\n`));
  process.exit(1);
}

async function findApiKey(provider = 'google') {
  const config = await readConfig();
  if (config.providers?.[provider]?.apiKey) return config.providers[provider].apiKey;
  const preset = getProviderPreset(provider);
  if (preset.env && process.env[preset.env]) return process.env[preset.env];
  return null;
}

async function ensureApiKeyForInteractive(config) {
  const existingKey = await findApiKey('google');
  if (existingKey) return existingKey;

  console.log(`\n${getLogo()}\n`);
  console.log(boxen(
    `${chalk.yellow('API ключ не найден.')}\n\n` +
    `Чтобы начать чат, сохраните ключ Gemini API.\n` +
    `${chalk.gray('Его можно получить в Google AI Studio.')}`,
    { padding: 1, borderColor: 'yellow', borderStyle: 'round', title: 'FIRST RUN' }
  ));

  const { apiKey } = await inquirer.prompt([{
    type: 'password',
    name: 'apiKey',
    message: 'Введите Gemini API key:',
    mask: '*',
    validate: value => value && value.trim().length > 0 ? true : 'Ключ не может быть пустым'
  }]);

  config.providers.google.apiKey = apiKey.trim();
  await writeConfig(config);
  console.log(chalk.green('✓ API ключ сохранён'));
  return config.providers.google.apiKey;
}

async function parseInputArgs(args) {
  const promptParts = [];
  const filesToProcess = [];

  for (const arg of args) {
    try {
      const matchedFiles = await glob(arg, { nodir: true, windowsPathsNoEscape: true });

      if (matchedFiles.length > 0) {
        filesToProcess.push(...matchedFiles);
        continue;
      }

      const stats = await fs.stat(arg);
      if (stats.isFile()) {
        filesToProcess.push(arg);
        continue;
      }
    } catch {}

    promptParts.push(arg);
  }

  return {
    prompt: promptParts.join(" ").trim(),
    filesToProcess
  };
}

function sanitizeFilePath(rawPath) {
  const target = path.resolve(rawPath);
  const relative = path.relative(process.cwd(), target);
  if (relative.startsWith('..') || path.isAbsolute(relative)) return null;
  return target;
}

async function readStdin() {
  if (process.stdin.isTTY) return null;
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  return Buffer.concat(chunks).toString("utf8");
}

function getMimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const mimeTypes = {
    '.png': 'image/png',
    '.jpeg': 'image/jpeg',
    '.jpg': 'image/jpeg',
    '.webp': 'image/webp',
    '.heic': 'image/heic',
    '.heif': 'image/heif',
    '.pdf': 'application/pdf',
    '.js': 'text/plain', // For LLMs, mostly text/plain is safe for code
    '.html': 'text/html',
    '.css': 'text/css',
    '.json': 'application/json',
    '.md': 'text/markdown',
    '.csv': 'text/csv'
  };
  return mimeTypes[ext] || 'text/plain';
}

async function showStats(modelKey = null) {
  try {
    let stats = await getUsageSnapshot();
    
    console.log(chalk.magenta.bold(`\n📊 СТАТИСТИКА ИСПОЛЬЗОВАНИЯ`));
    console.log(chalk.gray(`Дата: ${stats.lastReset}\n`));

    if (modelKey && stats.tokens[modelKey] !== undefined) {
      // Статистика по одной модели
      const tokens = stats.tokens[modelKey] || 0;
      const requests = stats.requests[modelKey] || 0;
      const tokenPrice = getTokenPrice(modelKey);
      const cost = (tokens / 1000) * tokenPrice;
      
      console.log(chalk.cyan.bold(`${modelKey.toUpperCase()}:`));
      console.log(chalk.gray(`  Запросов:     ${requests}`));
      console.log(chalk.gray(`  Токенов:      ${tokens}`));
      console.log(chalk.gray(`  Ср. на запрос: ${requests > 0 ? Math.round(tokens / requests) : 0} токенов`));
      console.log(chalk.gray(`  Стоимость:    ~$${cost.toFixed(4)}`));
    } else if (modelKey) {
      console.log(chalk.yellow(`Модель "${modelKey}" ещё не использована.`));
    } else {
      // Полная статистика за день
      let totalTokens = 0;
      let totalRequests = 0;
      let totalCost = 0;
      let hasData = false;

      const models = Object.keys(MODELS_CONFIG);
      for (const model of models) {
        const tokens = stats.tokens[model] || 0;
        const requests = stats.requests[model] || 0;
        
        if (requests === 0) continue;
        hasData = true;
        
        const tokenPrice = getTokenPrice(model);
        const cost = (tokens / 1000) * tokenPrice;
        
        totalTokens += tokens;
        totalRequests += requests;
        totalCost += cost;

        console.log(chalk.cyan(`${model.padEnd(10)} │ ${requests.toString().padEnd(3)} запросов │ ${tokens.toString().padEnd(6)} токенов │ $${cost.toFixed(4)}`));
        console.log(chalk.gray(`${' '.repeat(10)} │ осталось: ${formatRemainingLimit(model, requests)}`));
      }

      if (!hasData) {
        console.log(chalk.yellow("Нет данных. Сделайте первый запрос с помощью: gemini \"запрос\""));
      } else {
        console.log(chalk.gray(`─`.repeat(60)));
        console.log(chalk.yellow.bold(`ИТОГО:     │ ${totalRequests.toString().padEnd(3)} запросов │ ${totalTokens.toString().padEnd(6)} токенов │ $${totalCost.toFixed(4)}`));
      }
    }
    console.log();
  } catch (error) {
    console.error(chalk.red("❌ Ошибка:"), error.message || "Статистика не найдена. Сделайте первый запрос.");
  }
}

async function readFileAsPart(filePath) {
  const stats = await fs.stat(filePath);
  if (stats.size > MAX_FILE_SIZE) {
    console.warn(chalk.yellow(`⚠️  Пропущен файл больше ${Math.round(MAX_FILE_SIZE / 1024 / 1024)} МБ: ${filePath}`));
    return null;
  }
  const data = await fs.readFile(filePath);
  return { inlineData: { data: data.toString("base64"), mimeType: getMimeType(filePath) }};
}

async function readFileAsOpenAIContent(filePath) {
  const stats = await fs.stat(filePath);
  if (stats.size > MAX_FILE_SIZE) {
    console.warn(chalk.yellow(`⚠️  Пропущен файл больше ${Math.round(MAX_FILE_SIZE / 1024 / 1024)} МБ: ${filePath}`));
    return [];
  }

  const mimeType = getMimeType(filePath);
  const data = await fs.readFile(filePath);
  if (mimeType.startsWith('image/')) {
    return [{
      type: "image_url",
      image_url: { url: `data:${mimeType};base64,${data.toString("base64")}` }
    }];
  }

  return [{
    type: "text",
    text: `\n\n--- FILE: ${filePath} ---\n${data.toString("utf8")}`
  }];
}

async function buildFileParts(files) {
  const fileParts = [];
  const uniqueFiles = [...new Set(files)];
  for (const filePath of uniqueFiles) {
    try {
      const part = await readFileAsPart(filePath);
      if (part) fileParts.push(part);
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Не удалось прочитать файл ${filePath}: ${error.message}`));
    }
  }
  return fileParts;
}

function normalizeMessagePartsForOpenAI(messageParts) {
  if (typeof messageParts === 'string') return [{ type: "text", text: messageParts }];
  if (!Array.isArray(messageParts)) return [{ type: "text", text: String(messageParts || '') }];

  const content = [];
  for (const part of messageParts) {
    if (typeof part === 'string') {
      content.push({ type: "text", text: part });
      continue;
    }
    if (part?.openaiContent) {
      content.push(...part.openaiContent);
      continue;
    }
    if (part?.inlineData) {
      const { data, mimeType } = part.inlineData;
      if (mimeType?.startsWith('image/')) {
        content.push({ type: "image_url", image_url: { url: `data:${mimeType};base64,${data}` } });
      } else {
        content.push({ type: "text", text: `\n\n[base64 ${mimeType || 'file'} omitted for OpenAI-compatible provider]` });
      }
    }
  }
  return content.length > 0 ? content : [{ type: "text", text: "" }];
}

async function buildProviderFileParts(files, provider = 'google') {
  if (getProviderType(provider) !== 'openai-compatible') return buildFileParts(files);

  const fileParts = [];
  const uniqueFiles = [...new Set(files)];
  for (const filePath of uniqueFiles) {
    try {
      const openaiContent = await readFileAsOpenAIContent(filePath);
      if (openaiContent.length > 0) fileParts.push({ openaiContent });
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Не удалось прочитать файл ${filePath}: ${error.message}`));
    }
  }
  return fileParts;
}

function buildGenerationConfig(saveFiles, dryRun, config) {
  const generationConfig = { maxOutputTokens: config.maxOutputTokens || DEFAULT_SETTINGS.maxOutputTokens };

  if (saveFiles || dryRun) {
    generationConfig.responseMimeType = "application/json";
    generationConfig.responseSchema = {
      type: SchemaType.OBJECT,
      properties: {
        text: { type: SchemaType.STRING, description: "Ответ для пользователя (текст в формате Markdown)" },
        files: {
          type: SchemaType.ARRAY,
          description: "Список файлов для создания (оставь пустым, если файлы не требуются)",
          items: {
            type: SchemaType.OBJECT,
            properties: {
              path: { type: SchemaType.STRING, description: "Относительный путь к файлу" },
              content: { type: SchemaType.STRING, description: "Содержимое файла (исходный код)" }
            },
            required: ["path", "content"]
          }
        }
      },
      required: ["text", "files"]
    };
  }

  return generationConfig;
}

function buildSystemInstruction(saveFiles, dryRun) {
  if (saveFiles || dryRun) {
    return `${systemInstructionBase}\nВАЖНО: Всегда возвращай валидный JSON. Твой текстовый ответ помести в поле "text". Если пользователь просит создать файлы, добавь их в массив "files". Если файлы не нужны, массив "files" должен быть пустым.`;
  }
  return `${systemInstructionBase}\nВыводи текст как обычно (Markdown). Файлы сохранять не требуется.`;
}

function getOpenAIResponseSchema() {
  return {
    type: "json_schema",
    json_schema: {
      name: "gemini_cli_file_response",
      strict: true,
      schema: {
        type: "object",
        properties: {
          text: { type: "string" },
          files: {
            type: "array",
            items: {
              type: "object",
              properties: {
                path: { type: "string" },
                content: { type: "string" }
              },
              required: ["path", "content"],
              additionalProperties: false
            }
          }
        },
        required: ["text", "files"],
        additionalProperties: false
      }
    }
  };
}

function buildOpenAICompatibleRequest(modelCfg, providerConfig, session, messageParts, stream) {
  const messages = [
    { role: "system", content: buildSystemInstruction(session.saveFiles, session.dryRun) },
    { role: "user", content: normalizeMessagePartsForOpenAI(messageParts) }
  ];
  const body = {
    model: providerConfig.model || modelCfg.id,
    messages,
    stream,
    max_tokens: providerConfig.maxOutputTokens || session.maxOutputTokens || DEFAULT_SETTINGS.maxOutputTokens
  };

  if ((session.saveFiles || session.dryRun) && modelCfg.supportsStructuredOutput !== false) {
    body.response_format = getOpenAIResponseSchema();
  }

  return body;
}

function createOpenAICompatibleModel(modelKey, session, config) {
  const modelCfg = MODELS_CONFIG[modelKey];
  const provider = modelCfg.provider || 'openaiCompatible';
  const providerConfig = getProviderConfig(config, provider);
  if (!providerConfig.apiKey) throw new Error(`API ключ для провайдера "${provider}" не задан. Откройте /settings → API ключи провайдеров.`);
  if (!providerConfig.baseURL) throw new Error(`Base URL для провайдера "${provider}" не задан. Откройте /settings → API ключи провайдеров.`);

  return {
    startChat() {
      return {
        async sendMessage(messageParts) {
          const data = await callOpenAICompatible(providerConfig, buildOpenAICompatibleRequest(modelCfg, providerConfig, session, messageParts, false));
          return {
            response: {
              text: () => data.choices?.[0]?.message?.content || "",
              usageMetadata: {
                totalTokenCount: data.usage?.total_tokens || 0,
                promptTokenCount: data.usage?.prompt_tokens || 0,
                candidatesTokenCount: data.usage?.completion_tokens || 0
              }
            }
          };
        },
        async sendMessageStream(messageParts) {
          return callOpenAICompatibleStream(providerConfig, buildOpenAICompatibleRequest(modelCfg, providerConfig, session, messageParts, true));
        }
      };
    }
  };
}

function normalizeOpenAIBaseURL(baseURL) {
  return String(baseURL || '').replace(/\/+$/, '');
}

async function callOpenAICompatible(providerConfig, body) {
  const res = await fetch(`${normalizeOpenAIBaseURL(providerConfig.baseURL)}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${providerConfig.apiKey}`
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`[${res.status} ${res.statusText}] ${await res.text()}`);
  return res.json();
}

async function callOpenAICompatibleStream(providerConfig, body) {
  const res = await fetch(`${normalizeOpenAIBaseURL(providerConfig.baseURL)}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${providerConfig.apiKey}`
    },
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`[${res.status} ${res.statusText}] ${await res.text()}`);

  let responseResolver;
  const response = new Promise(resolve => { responseResolver = resolve; });
  async function* stream() {
    const decoder = new TextDecoder();
    let buffer = "";
    let responseText = "";
    let usageMetadata = { totalTokenCount: 0, promptTokenCount: 0, candidatesTokenCount: 0 };

    for await (const chunk of res.body) {
      buffer += decoder.decode(chunk, { stream: true });
      const lines = buffer.split(/\r?\n/);
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        let data;
        try {
          data = JSON.parse(payload);
        } catch {
          continue;
        }
        const usage = data.usage;
        if (usage) {
          usageMetadata = {
            totalTokenCount: usage.total_tokens || usageMetadata.totalTokenCount,
            promptTokenCount: usage.prompt_tokens || usageMetadata.promptTokenCount,
            candidatesTokenCount: usage.completion_tokens || usageMetadata.candidatesTokenCount
          };
        }
        const text = data.choices?.[0]?.delta?.content || "";
        if (text) {
          responseText += text;
          yield { text: () => text };
        }
      }
    }

    responseResolver({
      text: () => responseText,
      usageMetadata
    });
  }

  return { stream: stream(), response };
}

function createModel(genAI, modelKey, session, config) {
  modelKey = resolveModelKey(modelKey);
  const modelCfg = MODELS_CONFIG[modelKey];
  if (getProviderType(modelCfg.provider || 'google') === 'openai-compatible') {
    return createOpenAICompatibleModel(modelKey, session, config);
  }

  const modelOptions = {
    model: modelCfg.id,
    systemInstruction: buildSystemInstruction(session.saveFiles, session.dryRun),
    generationConfig: buildGenerationConfig(session.saveFiles, session.dryRun, config)
  };

  if (session.searchEnabled && modelCfg.supportsSearch !== false) {
    modelOptions.tools = [{ google_search: {} }];
  }

  return genAI.getGenerativeModel(modelOptions);
}

async function renderResponse(resultOrStream, session, modelKey) {
  let responseText = "";
  let usageTokenCount = 0;
  let promptTokenCount = '?';
  let candidatesTokenCount = '?';
  const provider = MODELS_CONFIG[modelKey]?.provider || 'google';
  const label = getProviderPreset(provider).label.split(/\s+/)[0].toUpperCase();

  if (session.saveFiles || session.dryRun) {
    const responseTextRaw = resultOrStream.response.text();
    const usage = resultOrStream.response.usageMetadata;
    usageTokenCount = usage?.totalTokenCount || 0;
    promptTokenCount = usage?.promptTokenCount || '?';
    candidatesTokenCount = usage?.candidatesTokenCount || '?';
    try {
      const parsed = JSON.parse(responseTextRaw);
      responseText = parsed.text || "";
      console.log(boxen(marked(responseText), { padding: 1, borderColor: 'magenta', borderStyle: 'round' }));
      await processSavedFiles(parsed.files, session.dryRun);
    } catch {
      responseText = responseTextRaw;
      console.log(marked(responseText) + "\n");
    }
  } else {
    process.stdout.write(chalk.bgMagenta.black.bold(` ${label} `) + " ");
    for await (const chunk of resultOrStream.stream) {
      const chunkText = chunk.text();
      responseText += chunkText;
      process.stdout.write(chunkText);
    }
    console.log("\n");
    const response = await resultOrStream.response;
    const usage = response.usageMetadata;
    usageTokenCount = usage?.totalTokenCount || 0;
    promptTokenCount = usage?.promptTokenCount || '?';
    candidatesTokenCount = usage?.candidatesTokenCount || '?';
  }

  return { responseText, usageTokenCount, promptTokenCount, candidatesTokenCount };
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function getErrorStatus(error) {
  const message = error?.message || String(error);
  const match = message.match(/\b(429|500|502|503)\b/);
  return match ? Number(match[1]) : null;
}

function isRetryableError(error) {
  const status = getErrorStatus(error);
  return status ? RETRY_CODES.includes(status) : false;
}

async function sendChatMessage(chat, messageParts, session, modelKey, { suppressError = false } = {}) {
  const spinner = ora({ text: chalk.magenta('Думаю...'), color: 'magenta' }).start();
  for (let attempt = 0; attempt <= RETRIES; attempt += 1) {
    try {
      const result = session.saveFiles || session.dryRun
        ? await chat.sendMessage(messageParts)
        : await chat.sendMessageStream(messageParts);
      spinner.stop();
      const usage = await renderResponse(result, session, modelKey);
      const stats = await updateAndGetUsage(modelKey, usage.usageTokenCount);
      return { ok: true, ...usage, stats };
    } catch (error) {
      const canRetry = isRetryableError(error) && attempt < RETRIES;
      if (canRetry) {
        const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
        spinner.text = chalk.yellow(`Временный сбой ${getErrorStatus(error) || ''}, повтор ${attempt + 1}/${RETRIES}...`);
        await sleep(delay);
        continue;
      }
      if (suppressError) {
        spinner.stop();
      } else {
        spinner.fail(chalk.red('Ошибка'));
      }
      if (!suppressError) console.error(chalk.red("❌:"), formatModelError(error));
      return {
        ok: false,
        quotaExceeded: isQuotaError(error),
        modelUnavailable: isModelUnavailableError(error),
        error
      };
    }
  }
}

async function sendWithAutoFallback({ createChatForModel, messageParts, makeMessageParts, session, modelKey, config }) {
  let currentModelKey = modelKey;
  const initialModelKey = modelKey;
  const triedModels = [];

  while (true) {
    const chat = await createChatForModel(currentModelKey);
    const currentMessageParts = makeMessageParts ? await makeMessageParts(currentModelKey) : messageParts;
    const result = await sendChatMessage(chat, currentMessageParts, session, currentModelKey, {
      suppressError: !!config.autoFallback
    });
    if (result?.ok) return { ...result, modelKey: currentModelKey };

    const shouldFallback = !!config.autoFallback && (result?.quotaExceeded || result?.modelUnavailable);
    if (!shouldFallback) {
      if (config.autoFallback && result?.error) {
        console.error(chalk.red("❌:"), formatModelError(result.error));
      }
      return { ...result, modelKey: initialModelKey };
    }

    const stats = await getUsageSnapshot();
    const nextModelKey = pickNextModel(currentModelKey, config, stats, triedModels);
    if (!nextModelKey) {
      console.error(chalk.red("❌:"), formatModelError(result.error));
      const chainText = getFallbackChain(config).join(' → ') || 'пусто';
      console.log(chalk.yellow(`Автозамена проверила цепочку (${chainText}) и не нашла доступную модель.`));
      return { ...result, modelKey: initialModelKey };
    }

    const reason = result?.quotaExceeded ? 'исчерпан' : 'недоступен';
    console.log(chalk.yellow(`↪ ${currentModelKey} ${reason} → перешёл на ${nextModelKey}`));
    triedModels.push(currentModelKey);
    currentModelKey = nextModelKey;
  }
}

function isQuotaError(error) {
  const message = (error?.message || String(error)).toLowerCase();
  return message.includes('429') || message.includes('too many requests') || message.includes('quota') || message.includes('rate-limit') || message.includes('rate limit');
}

function isModelUnavailableError(error) {
  const message = (error?.message || String(error)).toLowerCase();
  return message.includes('403') ||
    message.includes('404') ||
    message.includes('permission') ||
    message.includes('forbidden') ||
    message.includes('denied access') ||
    message.includes('not supported') ||
    message.includes('unsupported') ||
    message.includes('not found');
}

function formatModelError(error) {
  const message = error?.message || String(error);
  const lower = message.toLowerCase();
  const tips = [];

  if (isQuotaError(error)) {
    tips.push('лимит этой модели исчерпан; переключитесь через /model или дождитесь сброса квоты');
  }
  if (lower.includes('fetch failed') || lower.includes('fetch failled')) {
    tips.push('проверьте интернет/VPN/доступ к generativelanguage.googleapis.com');
  }
  if (lower.includes('not found') || lower.includes('404') || lower.includes('permission') || lower.includes('403')) {
    tips.push('модель недоступна проекту; выполните /sync-models или gemini --sync-models, чтобы скрыть такие модели из меню');
  }
  if (lower.includes('google_search') || lower.includes('tool')) {
    tips.push('попробуйте /search off');
  }
  if (lower.includes('api key') || lower.includes('401') || lower.includes('unauthorized')) {
    tips.push('проверьте ключ через /settings или gemini --doctor');
  }

  return tips.length > 0 ? `${message}\n${chalk.gray('Подсказка: ' + tips.join('; ') + '.')}` : message;
}

async function runDoctor() {
  const config = await readConfig();
  const statsExists = await pathExists(STATS_FILE) || await pathExists(LEGACY_STATS_FILE);
  const apiKey = await findApiKey();
  const checks = [
    ['Node.js', process.version],
    ['Папка конфигурации', APP_CONFIG_DIR],
    ['Google API ключ', config.providers?.google?.apiKey ? 'найден в конфиге' : (process.env.GEMINI_API_KEY ? 'найден в GEMINI_API_KEY' : 'не найден')],
    ['HTTPS_PROXY/HTTP_PROXY', process.env.HTTPS_PROXY || process.env.HTTP_PROXY || process.env.https_proxy || process.env.http_proxy || 'не задан'],
    ['Модель по умолчанию', resolveModelKey(config.defaultModel || 'flash')],
    ['Статистика', statsExists ? 'найдена' : 'пока нет данных']
  ];

  console.log(chalk.magenta.bold('\n🩺 GEMINI CLI DOCTOR\n'));
  for (const [name, value] of checks) {
    const ok = !String(value).includes('не найден');
    console.log(`${ok ? chalk.green('✓') : chalk.yellow('!')} ${chalk.bold(name)}: ${value}`);
  }

  const modules = ['@google/generative-ai', 'commander', 'glob', 'inquirer', 'marked'];
  for (const moduleName of modules) {
    try {
      require.resolve(moduleName);
      console.log(`${chalk.green('✓')} ${chalk.bold(moduleName)} установлен`);
    } catch {
      console.log(`${chalk.red('✗')} ${chalk.bold(moduleName)} не установлен`);
    }
  }

  if (apiKey) {
    const spinner = ora(chalk.cyan('Проверяю доступ к Google Generative Language API...')).start();
    try {
      const catalog = await fetchAvailableModels(apiKey);
      const validation = config.validateModelsOnSync
        ? await validateModelCatalog(apiKey, catalog, spinner)
        : { validated: catalog, denied: [] };
      spinner.succeed(chalk.green(`API доступен, usable моделей: ${Object.keys(validation.validated).length}, скрыто: ${validation.denied.length}`));
    } catch (error) {
      spinner.fail(chalk.red(`API недоступен: ${error.message}`));
      console.log(chalk.gray('Если это fetch failed, проверьте интернет, VPN/proxy, DNS и доступ к generativelanguage.googleapis.com из Node.js.'));
    }
  }

  if (await pathExists(LEGACY_CONFIG_FILE) || await pathExists(LEGACY_STATS_FILE)) {
    console.log(chalk.gray(`\nНайдены старые файлы в папке проекта. CLI их читает для совместимости, новые записи идут в ${CONFIG_FILE}.`));
  }
  console.log();
}

async function run() {
  program
    .name('gemini')
    .description(gradient.mind("🤖 GEMINI CLI HELPER — Мощный помощник в вашем терминале"))
    .usage('[запрос и файлы] [флаги]')
    .option('-m, --model <key>', 'Использовать конкретную модель')
    .option('-c, --config', 'Выбрать и СОХРАНИТЬ модель по умолчанию')
    .option('-i, --interactive', 'Интерактивный режим чата')
    .option('--save', 'Разрешить сохранение файлов (Structured Outputs)')
    .option('-d, --dry-run', 'Режим симуляции (dry-run)')
    .option('--set-key <key>', 'Сохранить API ключ')
    .option('--scan', 'Показать доступные модели')
    .option('--sync-models', 'Синхронизировать модели, доступные по API ключу')
    .option('--models', 'Показать локально настроенные модели')
    .option('--providers', 'Показать провайдеры, ссылки на API ключи и статус подключения')
    .option('--stats [model]', 'Показать статистику использования')
    .option('--reset-stats', 'Сбросить локальную статистику')
    .option('--config-path', 'Показать пути к конфигу и статистике')
    .option('--settings', 'Открыть интерактивные настройки')
    .option('--doctor', 'Проверить окружение и настройки CLI')
    .helpOption('-h, --help', 'Показать справку')
    .addHelpText('before', `\n${gradient.fruit.multiline(
      "  ____ _____ __  __ ___ _   _ ___ \n" +
      " / ___| ____|  \\/  |_ _| \\ | |_ _| \n" +
      "| |  _|  _| | |\\/| || ||  \\| || |  \n" +
      "| |_| | |___| |  | || || |\\  || |  \n" +
      " \\____|_____|_|  |_|___|_| \\_|___| \n"
    )}\n`)
    .addHelpText('after', `
${chalk.cyan.bold("Доступные модели:")}
  ${chalk.yellow('lite'.padEnd(12))} → gemini-3.1-flash-lite (Google, лимит 500/день)
  ${chalk.yellow('gemma'.padEnd(12))} → gemma-4-31b-it (Google, высокий лимит)
  ${chalk.yellow('deepseek'.padEnd(12))} → deepseek-v4-flash (DeepSeek, нужен ключ)
  ${chalk.yellow('qwen'.padEnd(12))} → qwen-plus (Qwen/DashScope, нужен ключ)

${chalk.cyan.bold("Примеры:")}
  ${chalk.green("gemini")} "Как дела?"
  ${chalk.green("gemini")} "Проверь код" index.js ${chalk.yellow("--save")}
  ${chalk.green("gemini")} ${chalk.yellow("--providers")}
  ${chalk.green("gemini")} ${chalk.yellow("--stats")}
  ${chalk.green("gemini")} ${chalk.yellow("--models")}
  ${chalk.green("gemini")} ${chalk.yellow("-i")}
`);

  program.parse(process.argv);
  const options = program.opts();
  const positional = program.args;
  let config = await readConfig();

  let dryRun = options.dryRun ? true : !!config.dryRunByDefault;
  let saveFiles = options.save ? true : !!config.saveByDefault;
  let interactiveMode = !!options.interactive;
  let selectedModelKey = await getDefaultModel();

  if (options.setKey) {
    config.providers.google.apiKey = options.setKey;
    await writeConfig(config);
    console.log(chalk.green(`✅ Google API ключ успешно сохранён!`));
    if (positional.length === 0 && !interactiveMode) return;
  }

  if (options.scan) { await scanModels(); return; }
  if (options.syncModels) { await syncModelsCommand(config); return; }
  if (options.models) { showConfiguredModels(); return; }
  if (options.providers) { showProvidersGuide(config); return; }
  if (options.stats !== undefined) {
    await showStats(options.stats === true ? null : options.stats);
    return;
  }
  if (options.resetStats) { await resetStats(); return; }
  if (options.configPath) { showConfigPath(); return; }
  if (options.settings) { await openSettingsMenu(config); return; }
  if (options.doctor) { await runDoctor(); return; }

  if (options.config) {
    const { modelKey } = await inquirer.prompt([{
      type: 'list', name: 'modelKey', message: 'Выберите модель по умолчанию:',
      choices: Object.keys(MODELS_CONFIG).map(k => ({
        name: `${chalk.bold(k.padEnd(8))} | ${MODELS_CONFIG[k].desc}`, value: k
      }))
    }]);
    selectedModelKey = modelKey;
    await setDefaultModel(modelKey);
    console.log(chalk.green(`✅ Модель `) + chalk.bold.cyan(selectedModelKey) + chalk.green(` сохранена как модель по умолчанию.\n`));
    if (positional.length === 0 && !interactiveMode) return;
  }

  if (options.model) { selectedModelKey = resolveModelKey(options.model); }

  const stdinContent = await readStdin();
  if (positional.length === 0 && !stdinContent) {
    interactiveMode = true;
  }

  const parsedInput = await parseInputArgs(positional);
  const prompt = [parsedInput.prompt, stdinContent].filter(Boolean).join("\n\n").trim();
  const filesToProcess = parsedInput.filesToProcess;

  let apiKey = interactiveMode ? await ensureApiKeyForInteractive(config) : await getApiKey();
  if (config.autoSyncModels) {
    try {
      await syncModels(apiKey, config, { force: false, silent: false });
      config = await readConfig();
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Не удалось синхронизировать модели: ${error.message}`));
      console.warn(chalk.gray('Продолжаю с локальным списком. Для ручной проверки используйте gemini --sync-models.'));
    }
  }
  let genAI = new GoogleGenerativeAI(apiKey);

  selectedModelKey = resolveModelKey(selectedModelKey);
  const usageSnapshot = await getUsageSnapshot();
  const requestedModelKey = selectedModelKey;
  selectedModelKey = pickInitialModel(selectedModelKey, config, usageSnapshot);
  if (requestedModelKey !== selectedModelKey) {
    console.log(chalk.yellow(`↪ ${requestedModelKey} уже у локального дневного лимита → стартую с ${selectedModelKey}`));
  }
  showLimitWarning(selectedModelKey, usageSnapshot);
  const session = {
    saveFiles,
    dryRun,
    searchEnabled: !!config.searchEnabled,
    maxOutputTokens: config.maxOutputTokens || DEFAULT_SETTINGS.maxOutputTokens,
    pendingFiles: []
  };

  let model = createModel(genAI, selectedModelKey, session, config);

  if (interactiveMode) {
    let chat = model.startChat({ history: [] });

    const recreateChat = () => {
      model = createModel(genAI, selectedModelKey, session, config);
      chat = model.startChat({ history: [] });
    };

    const switchSessionModel = async (nextModelKey) => {
      selectedModelKey = nextModelKey;
      recreateChat();
    };

    const createChatForModel = async (nextModelKey) => {
      if (nextModelKey !== selectedModelKey) await switchSessionModel(nextModelKey);
      return chat;
    };

    await showHomeScreen(selectedModelKey, config, session);

    if (prompt || filesToProcess.length > 0) {
      console.log(chalk.bgBlue.black.bold(" ВЫ ") + " " + (prompt || chalk.gray("(только файлы)")));
      const result = await sendWithAutoFallback({
        createChatForModel,
        makeMessageParts: async (nextModelKey) => {
          const provider = MODELS_CONFIG[nextModelKey]?.provider || 'google';
          const fileParts = await buildProviderFileParts(filesToProcess, provider);
          return filesToProcess.length > 0 ? [prompt, ...fileParts] : prompt;
        },
        session,
        modelKey: selectedModelKey,
        config
      });
      selectedModelKey = result.modelKey || selectedModelKey;
      if (result?.quotaExceeded && !config.autoFallback) {
        selectedModelKey = await offerQuotaModelSwitch(selectedModelKey, switchSessionModel);
      }
    }

    while (true) {
      const { userPrompt } = await inquirer.prompt([{
        type: 'input',
        name: 'userPrompt',
        prefix: '',
        message: `${chalk.bgBlue.black.bold(" YOU ")} ${chalk.gray(`[${selectedModelKey} · ${session.saveFiles ? 'save' : 'chat'}${session.dryRun ? ' · dry' : ''}${session.searchEnabled ? ' · search' : ''}]`)}`,
      }]);

      if (!userPrompt || userPrompt.trim() === '') continue;
      const trimmedPrompt = normalizeChatCommand(userPrompt);
      const lowerPrompt = trimmedPrompt.toLowerCase();

      if (lowerPrompt === 'exit' || lowerPrompt === 'quit' || lowerPrompt === '/exit' || lowerPrompt === '/quit' || lowerPrompt === '/q') {
        console.log(chalk.gray('Завершение чата.'));
        break;
      }

      if (lowerPrompt.startsWith('/')) {
        const [command, ...commandArgs] = trimmedPrompt.split(/\s+/);
        const commandArgText = commandArgs.join(" ");

        if (command === '/help' || command === '/h' || command === '/?') {
          showChatHelp();
          continue;
        }

        if (command === '/model' || command === '/m') {
          if (commandArgs[0] && MODELS_CONFIG[commandArgs[0]]) {
            selectedModelKey = commandArgs[0];
          } else {
            if (commandArgs[0]) console.log(chalk.yellow(`Модель "${commandArgs[0]}" не найдена.`));
            selectedModelKey = await promptModelChoice(selectedModelKey);
          }
          const { saveDefault } = await inquirer.prompt([{
            type: 'confirm',
            name: 'saveDefault',
            message: 'Сохранить эту модель по умолчанию?',
            default: false
          }]);
          if (saveDefault) {
            config.defaultModel = selectedModelKey;
            await writeConfig(config);
          }
          recreateChat();
          console.log(chalk.green(`✓ Модель сессии: ${selectedModelKey}. История чата очищена.`));
          continue;
        }

        if (command === '/settings' || command === '/config') {
          config = await openSettingsMenu(config);
          const nextApiKey = await findApiKey('google');
          if (nextApiKey && nextApiKey !== apiKey) {
            apiKey = nextApiKey;
            genAI = new GoogleGenerativeAI(apiKey);
          }
          selectedModelKey = resolveModelKey(config.defaultModel || selectedModelKey);
          session.saveFiles = !!config.saveByDefault;
          session.dryRun = !!config.dryRunByDefault;
          session.searchEnabled = !!config.searchEnabled;
          session.maxOutputTokens = config.maxOutputTokens || DEFAULT_SETTINGS.maxOutputTokens;
          recreateChat();
          console.log(chalk.green('✓ Настройки применены. История чата очищена.'));
          continue;
        }

        if (command === '/stats') {
          await showStats(commandArgs[0] || null);
          continue;
        }

        if (command === '/models') {
          if (commandArgs.includes('--refresh') || commandArgs.includes('-r')) {
            try {
              await syncModels(apiKey, config, { force: true });
              config = await readConfig();
              selectedModelKey = resolveModelKey(selectedModelKey);
              recreateChat();
            } catch (error) {
              console.error(chalk.red("❌ Не удалось обновить модели:"), error.message);
            }
          }
          showConfiguredModels();
          showDeniedModels(config);
          continue;
        }

        if (command === '/providers') {
          showProvidersGuide(config);
          continue;
        }

        if (command === '/sync-models') {
          try {
            await syncModels(apiKey, config, { force: true });
            config = await readConfig();
            selectedModelKey = resolveModelKey(selectedModelKey);
            recreateChat();
            console.log(chalk.green('✓ Модели синхронизированы. История чата очищена.'));
            showDeniedModels(config);
          } catch (error) {
            console.error(chalk.red("❌ Не удалось обновить модели:"), error.message);
          }
          continue;
        }

        if (command === '/path' || command === '/config-path') {
          showConfigPath();
          continue;
        }

        if (command === '/doctor') {
          await runDoctor();
          continue;
        }

        if (command === '/reset-stats') {
          const { confirmReset } = await inquirer.prompt([{
            type: 'confirm',
            name: 'confirmReset',
            message: 'Сбросить локальную статистику?',
            default: false
          }]);
          if (confirmReset) await resetStats();
          continue;
        }

        if (command === '/save') {
          session.saveFiles = parseToggleArg(commandArgs[0], session.saveFiles);
          recreateChat();
          console.log(chalk.green(`✓ Save mode: ${session.saveFiles ? 'on' : 'off'}. История чата очищена.`));
          continue;
        }

        if (command === '/dry') {
          session.dryRun = parseToggleArg(commandArgs[0], session.dryRun);
          if (session.dryRun) session.saveFiles = true;
          recreateChat();
          console.log(chalk.green(`✓ Dry-run: ${session.dryRun ? 'on' : 'off'}. История чата очищена.`));
          continue;
        }

        if (command === '/search') {
          session.searchEnabled = parseToggleArg(commandArgs[0], session.searchEnabled);
          recreateChat();
          console.log(chalk.green(`✓ Google Search: ${session.searchEnabled ? 'on' : 'off'}. История чата очищена.`));
          continue;
        }

        if (command === '/clear' || command === '/new') {
          recreateChat();
          console.log(chalk.green('✓ История чата очищена.'));
          continue;
        }

        if (command === '/home') {
          await showHomeScreen(selectedModelKey, config, session);
          continue;
        }

        if (command === '/files') {
          if (!commandArgText) {
            console.log(chalk.yellow('Укажите файлы или glob: /files README.md "*.js"'));
            continue;
          }
          const parsedFiles = await parseInputArgs(commandArgs);
          session.pendingFiles.push(...parsedFiles.filesToProcess);
          console.log(chalk.green(`✓ Файлов к следующему сообщению: ${session.pendingFiles.length}`));
          if (parsedFiles.prompt) console.log(chalk.gray(`Не распознано как файл: ${parsedFiles.prompt}`));
          continue;
        }

        console.log(chalk.yellow(`Неизвестная команда: ${command}. Введите /help.`));
        continue;
      }

      const pendingFiles = session.pendingFiles;
      session.pendingFiles = [];
      const result = await sendWithAutoFallback({
        createChatForModel,
        makeMessageParts: async (nextModelKey) => {
          const provider = MODELS_CONFIG[nextModelKey]?.provider || 'google';
          const pendingFileParts = await buildProviderFileParts(pendingFiles, provider);
          return pendingFileParts.length > 0 ? [trimmedPrompt, ...pendingFileParts] : trimmedPrompt;
        },
        session,
        modelKey: selectedModelKey,
        config
      });
      selectedModelKey = result.modelKey || selectedModelKey;
      if (result?.quotaExceeded && !config.autoFallback) {
        selectedModelKey = await offerQuotaModelSwitch(selectedModelKey, switchSessionModel);
      }
    }
    return;
  }

  const result = await sendWithAutoFallback({
    createChatForModel: async (nextModelKey) => {
      selectedModelKey = nextModelKey;
      model = createModel(genAI, selectedModelKey, session, config);
      return model.startChat({ history: [] });
    },
    makeMessageParts: async (nextModelKey) => {
      const provider = MODELS_CONFIG[nextModelKey]?.provider || 'google';
      const fileParts = await buildProviderFileParts(filesToProcess, provider);
      return [prompt, ...fileParts];
    },
    session,
    modelKey: selectedModelKey,
    config
  });

  selectedModelKey = result.modelKey || selectedModelKey;
  if (!result.ok) return;

  const totalTokens = result.stats.tokens[selectedModelKey] || 0;
  const totalRequests = result.stats.requests[selectedModelKey] || 0;
  const tokenPricePer1k = getTokenPrice(selectedModelKey);
  const estimatedCost = (totalTokens / 1000) * tokenPricePer1k;

  const statsBox = boxen(
    `${chalk.cyan.bold(`📊 СТАТИСТИКА [${selectedModelKey}]`)}\n\n` +
    `${chalk.gray(`Запрос:       `)}${chalk.white(`${result.promptTokenCount} + ${result.candidatesTokenCount} = ${result.usageTokenCount} токенов`)}\n` +
    `${chalk.gray(`За день:      `)}${chalk.white(`${totalTokens} токенов (${totalRequests} запросов)`)}\n` +
    `${chalk.yellow.bold(`Стоимость:    ~$${estimatedCost.toFixed(4)}`)}`,
    { padding: 1, margin: { top: 1 }, borderStyle: 'round', borderColor: 'cyan' }
  );
  console.log(statsBox);
}

if (require.main === module) {
  run();
}

module.exports = {
  getMimeType,
  normalizeStats,
  normalizeChatCommand,
  parseInputArgs,
  parseToggleArg,
  pickInitialModel,
  pickNextModel,
  normalizeConfig,
  isQuotaError,
  isModelUnavailableError,
  sanitizeFilePath
};
