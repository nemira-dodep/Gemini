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

let MODELS_CONFIG = {
  "flash": {
    id: "gemini-3-flash-preview",
    desc: "🚀 Основная модель. Скорость и новейшие знания 2026.",
    dailyLimit: 20
  },
  "lite": {
    id: "gemini-3.1-flash-lite",
    desc: "☁️ Высокие лимиты (500/день). Для рутины и тестов.",
    dailyLimit: 500
  },
  "gemma": {
    id: "gemma-4-31b-it",
    desc: "📟 Огромные лимиты. Для массовой обработки текста.",
    dailyLimit: 14400
  },
  "research": {
    id: "deep-research-pro-preview-12-2025",
    desc: "🔍 Глубокий поиск и создание документации.",
    dailyLimit: 500
  },
  "vision": {
    id: "gemini-3.1-flash-image-preview",
    desc: "👁️ Анализ скриншотов, макетов и графиков.",
    dailyLimit: 20
  }
};

const systemInstructionBase = `Ты — экспертный CLI-помощник.`;
const DEFAULT_SETTINGS = {
  defaultModel: "flash",
  saveByDefault: false,
  dryRunByDefault: false,
  searchEnabled: true,
  maxOutputTokens: 8192,
  showWelcome: true,
  autoSyncModels: true,
  validateModelsOnSync: true,
  modelsSyncedAt: null,
  availableModels: null
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

async function readConfig() {
  const config = { ...DEFAULT_SETTINGS, ...(await readAppJson(CONFIG_FILE, LEGACY_CONFIG_FILE, {})) };
  applyAvailableModels(config);
  return config;
}

async function writeConfig(config) {
  await writeAppJson(CONFIG_FILE, { ...DEFAULT_SETTINGS, ...config });
}

function applyAvailableModels(config) {
  if (config?.availableModels && Object.keys(config.availableModels).length > 0) {
    MODELS_CONFIG = config.availableModels;
  }
}

function getPreferredAlias(modelId, usedAliases) {
  const id = modelId.toLowerCase();
  const candidates = [];

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

    const alias = getPreferredAlias(id, usedAliases);
    usedAliases.add(alias);
    catalog[alias] = {
      id,
      desc: model.displayName || model.description || 'Доступна для generateContent по вашему API ключу',
      dailyLimit: MODELS_CONFIG[alias]?.dailyLimit || 0,
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
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:countTokens?key=${apiKey}`;
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ role: 'user', parts: [{ text: 'ping' }] }]
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
        throw new Error('Ни одна модель не прошла проверку доступа countTokens');
      }
    }

    config.availableModels = catalog;
    config.deniedModels = deniedModels;
    config.modelsSyncedAt = new Date().toISOString();
    if (!catalog[config.defaultModel]) {
      const preferred = catalog.flash ? 'flash' : Object.keys(catalog)[0];
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
  const fallback = MODELS_CONFIG.flash ? "flash" : Object.keys(MODELS_CONFIG)[0];
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

function showConfiguredModels() {
  console.log(chalk.magenta.bold('\n🤖 НАСТРОЕННЫЕ МОДЕЛИ\n'));
  for (const [key, cfg] of Object.entries(MODELS_CONFIG)) {
    console.log(`${chalk.yellow(key.padEnd(10))} ${chalk.cyan(cfg.id)}`);
    console.log(chalk.gray(`           ${cfg.desc}`));
    console.log(chalk.gray(`           Локальный лимит: ${cfg.dailyLimit ? `${cfg.dailyLimit}/день` : 'не задан'}${cfg.source === 'api' ? ' · synced' : ''}\n`));
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
  const remaining = formatRemainingLimit(modelKey, requests);
  const body = [
    `${chalk.bold("Model")}      ${chalk.yellow(modelKey)} ${chalk.gray(`(${cfg.id})`)}`,
    `${chalk.bold("Mode")}       ${getModeLabel(session.saveFiles, session.dryRun)} ${chalk.gray(session.searchEnabled ? "search:on" : "search:off")}`,
    `${chalk.bold("Today")}      ${requests} req · ${tokens} tokens · left ${remaining}`,
    `${chalk.bold("Config")}     ${CONFIG_FILE}`,
    "",
    `${chalk.gray("Slash commands:")} ${chalk.cyan("/model")} ${chalk.cyan("/settings")} ${chalk.cyan("/stats")} ${chalk.cyan("/save")} ${chalk.cyan("/dry")} ${chalk.cyan("/search")} ${chalk.cyan("/clear")} ${chalk.cyan("/help")} ${chalk.cyan("/exit")}`,
    `${chalk.gray("Files:")} type ${chalk.cyan('/files README.md "*.js"')} before your next message`
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
    `${chalk.cyan("/model, /m [key]")}    выбрать модель для этой сессии`,
    `${chalk.cyan("/settings, /config")}  открыть настройки CLI`,
    `${chalk.cyan("/stats [key]")}        показать статистику`,
    `${chalk.cyan("/models")}             показать локально настроенные модели`,
    `${chalk.cyan("/sync-models")}        обновить список моделей по API ключу`,
    `${chalk.cyan("/doctor")}             проверить окружение`,
    `${chalk.cyan("/path")}               показать пути к конфигу и статистике`,
    `${chalk.cyan("/reset-stats")}        сбросить локальную статистику`,
    `${chalk.cyan("/save [on|off]")}      переключить сохранение файлов`,
    `${chalk.cyan("/dry [on|off]")}       переключить dry-run`,
    `${chalk.cyan("/search [on|off]")}    включить/выключить Google Search`,
    `${chalk.cyan('/files <glob>')}        добавить файлы к следующему сообщению`,
    `${chalk.cyan("/clear")}              начать новый чат без истории`,
    `${chalk.cyan("/home")}               показать стартовый экран`,
    `${chalk.cyan("/exit, /q")}           выйти`
  ].join("\n"), { padding: 1, borderColor: 'cyan', borderStyle: 'round', title: 'COMMANDS' }));
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

async function openSettingsMenu(config) {
  let nextConfig = { ...DEFAULT_SETTINGS, ...config };

  while (true) {
    const { action } = await inquirer.prompt([{
      type: 'list',
      name: 'action',
      message: 'Настройки Gemini CLI',
      choices: [
        { name: `API ключ: ${process.env.GEMINI_API_KEY ? 'из GEMINI_API_KEY' : (nextConfig.apiKey ? 'сохранен в конфиге' : 'не задан')}`, value: 'key' },
        { name: `Модель по умолчанию: ${nextConfig.defaultModel}`, value: 'model' },
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
    if (action === 'key') {
      const { keyAction } = await inquirer.prompt([{
        type: 'list',
        name: 'keyAction',
        message: 'API ключ',
        choices: [
          { name: 'Сохранить новый ключ', value: 'set' },
          { name: 'Удалить ключ из config.json', value: 'clear' },
          { name: 'Назад', value: 'back' }
        ]
      }]);
      if (keyAction === 'set') {
        const { apiKey } = await inquirer.prompt([{
          type: 'password',
          name: 'apiKey',
          message: 'Введите Gemini API key:',
          mask: '*',
          validate: value => value && value.trim().length > 0 ? true : 'Ключ не может быть пустым'
        }]);
        nextConfig.apiKey = apiKey.trim();
      }
      if (keyAction === 'clear') {
        delete nextConfig.apiKey;
      }
      if (keyAction === 'back') continue;
    }
    if (action === 'model') nextConfig.defaultModel = await promptModelChoice(nextConfig.defaultModel);
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

async function getApiKey() {
  const key = await findApiKey();
  if (key) return key;

  console.error(chalk.red("\n❌ Ошибка: API ключ не найден."));
  console.log(chalk.gray(`Пожалуйста, установите ключ с помощью команды: ${chalk.cyan('gemini --set-key "ваш_ключ"')}\n`));
  process.exit(1);
}

async function findApiKey() {
  if (process.env.GEMINI_API_KEY) return process.env.GEMINI_API_KEY;
  const config = await readConfig();
  if (config.apiKey) return config.apiKey;
  return null;
}

async function ensureApiKeyForInteractive(config) {
  const existingKey = await findApiKey();
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

  config.apiKey = apiKey.trim();
  await writeConfig(config);
  console.log(chalk.green('✓ API ключ сохранён'));
  return config.apiKey;
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

function createModel(genAI, modelKey, session, config) {
  modelKey = resolveModelKey(modelKey);
  const modelCfg = MODELS_CONFIG[modelKey];
  const modelOptions = {
    model: modelCfg.id,
    systemInstruction: buildSystemInstruction(session.saveFiles, session.dryRun),
    generationConfig: buildGenerationConfig(session.saveFiles, session.dryRun, config)
  };

  if (session.searchEnabled) {
    modelOptions.tools = [{ google_search: {} }];
  }

  return genAI.getGenerativeModel(modelOptions);
}

async function renderResponse(resultOrStream, session) {
  let responseText = "";
  let usageTokenCount = 0;

  if (session.saveFiles || session.dryRun) {
    const responseTextRaw = resultOrStream.response.text();
    usageTokenCount = resultOrStream.response.usageMetadata?.totalTokenCount || 0;
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
    process.stdout.write(chalk.bgMagenta.black.bold(" GEMINI ") + " ");
    for await (const chunk of resultOrStream.stream) {
      const chunkText = chunk.text();
      responseText += chunkText;
      process.stdout.write(chunkText);
    }
    console.log("\n");
    const response = await resultOrStream.response;
    usageTokenCount = response.usageMetadata?.totalTokenCount || 0;
  }

  return { responseText, usageTokenCount };
}

async function sendChatMessage(chat, messageParts, session, modelKey) {
  const spinner = ora({ text: chalk.magenta('Думаю...'), color: 'magenta' }).start();
  try {
    const result = session.saveFiles || session.dryRun
      ? await chat.sendMessage(messageParts)
      : await chat.sendMessageStream(messageParts);
    spinner.stop();
    const { usageTokenCount } = await renderResponse(result, session);
    await updateAndGetUsage(modelKey, usageTokenCount);
    return { ok: true };
  } catch (error) {
    spinner.fail(chalk.red('Ошибка'));
    console.error(chalk.red("❌:"), formatModelError(error));
    return { ok: false, quotaExceeded: isQuotaError(error), error };
  }
}

function isQuotaError(error) {
  const message = (error?.message || String(error)).toLowerCase();
  return message.includes('429') || message.includes('too many requests') || message.includes('quota') || message.includes('rate-limit') || message.includes('rate limit');
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
    ['API ключ', process.env.GEMINI_API_KEY ? 'найден в GEMINI_API_KEY' : (config.apiKey ? 'найден в конфиге' : 'не найден')],
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
  ${chalk.yellow('flash'.padEnd(8))} → gemini-3-flash-preview (Основная, быстрая)
  ${chalk.yellow('lite'.padEnd(8))} → gemini-3.1-flash-lite (Лимит 500/день)
  ${chalk.yellow('gemma'.padEnd(8))} → gemma-4-31b-it (Лимит 14400/день)
  ${chalk.yellow('research'.padEnd(8))} → deep-research-pro (Глубокий поиск)
  ${chalk.yellow('vision'.padEnd(8))} → gemini-3.1-flash-image (Анализ изображений)

${chalk.cyan.bold("Примеры:")}
  ${chalk.green("gemini")} "Как дела?"
  ${chalk.green("gemini")} "Проверь код" index.js ${chalk.yellow("--save")}
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
    config.apiKey = options.setKey;
    await writeConfig(config);
    console.log(chalk.green(`✅ API ключ успешно сохранён!`));
    if (positional.length === 0 && !interactiveMode) return;
  }

  if (options.scan) { await scanModels(); return; }
  if (options.syncModels) { await syncModelsCommand(config); return; }
  if (options.models) { showConfiguredModels(); return; }
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

  const apiKey = interactiveMode ? await ensureApiKeyForInteractive(config) : await getApiKey();
  if (config.autoSyncModels) {
    try {
      await syncModels(apiKey, config, { force: false, silent: false });
      config = await readConfig();
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Не удалось синхронизировать модели: ${error.message}`));
      console.warn(chalk.gray('Продолжаю с локальным списком. Для ручной проверки используйте gemini --sync-models.'));
    }
  }
  const genAI = new GoogleGenerativeAI(apiKey);

  selectedModelKey = resolveModelKey(selectedModelKey);
  showLimitWarning(selectedModelKey, await getUsageSnapshot());
  const session = {
    saveFiles,
    dryRun,
    searchEnabled: !!config.searchEnabled,
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

    await showHomeScreen(selectedModelKey, config, session);

    if (prompt || filesToProcess.length > 0) {
      console.log(chalk.bgBlue.black.bold(" ВЫ ") + " " + (prompt || chalk.gray("(только файлы)")));
      const fileParts = await buildFileParts(filesToProcess);
      const result = await sendChatMessage(chat, filesToProcess.length > 0 ? [prompt, ...fileParts] : prompt, session, selectedModelKey);
      if (result?.quotaExceeded) {
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
          selectedModelKey = resolveModelKey(config.defaultModel || selectedModelKey);
          session.saveFiles = !!config.saveByDefault;
          session.dryRun = !!config.dryRunByDefault;
          session.searchEnabled = !!config.searchEnabled;
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

      const pendingFileParts = await buildFileParts(session.pendingFiles);
      const messageParts = pendingFileParts.length > 0 ? [trimmedPrompt, ...pendingFileParts] : trimmedPrompt;
      session.pendingFiles = [];
      const result = await sendChatMessage(chat, messageParts, session, selectedModelKey);
      if (result?.quotaExceeded) {
        selectedModelKey = await offerQuotaModelSwitch(selectedModelKey, switchSessionModel);
      }
    }
    return;
  }

  const spinner = ora({ text: chalk.magenta(`Gemini [${selectedModelKey}] на связи...`), color: 'magenta' }).start();

  try {
    const fileParts = await buildFileParts(filesToProcess);

    let responseText = "";
    let usageTokenCount = 0;
    let promptTokenCount = '?';
    let candidatesTokenCount = '?';

    if (saveFiles || dryRun) {
      const result = await model.generateContent([prompt, ...fileParts]);
      spinner.succeed(chalk.green('Готово!\n'));
      responseText = result.response.text();
      const usage = result.response.usageMetadata;
      usageTokenCount = usage?.totalTokenCount || 0;
      promptTokenCount = usage?.promptTokenCount || '?';
      candidatesTokenCount = usage?.candidatesTokenCount || '?';
      try {
        const parsed = JSON.parse(responseText);
        console.log(boxen(marked(parsed.text), { padding: 1, borderColor: 'magenta', borderStyle: 'round', title: 'GEMINI RESPONSE', titleAlignment: 'center' }));
        await processSavedFiles(parsed.files, dryRun);
      } catch(e) { console.log(marked(responseText) + "\n"); }
    } else {
      const result = await model.generateContentStream([prompt, ...fileParts]);
      spinner.succeed(chalk.green('Готово!\n'));
      process.stdout.write(chalk.bgMagenta.black.bold(" GEMINI ") + " ");
      for await (const chunk of result.stream) {
        const chunkText = chunk.text();
        responseText += chunkText;
        process.stdout.write(chunkText);
      }
      console.log("\n");
      const response = await result.response;
      const usage = response.usageMetadata;
      usageTokenCount = usage?.totalTokenCount || 0;
      promptTokenCount = usage?.promptTokenCount || '?';
      candidatesTokenCount = usage?.candidatesTokenCount || '?';
    }

    const stats = await updateAndGetUsage(selectedModelKey, usageTokenCount);
    const totalTokens = stats.tokens[selectedModelKey] || 0;
    const totalRequests = stats.requests[selectedModelKey] || 0;
    
    const tokenPricePer1k = getTokenPrice(selectedModelKey);
    const estimatedCost = (totalTokens / 1000) * tokenPricePer1k;

    const statsBox = boxen(
      `${chalk.cyan.bold(`📊 СТАТИСТИКА [${selectedModelKey}]`)}\n\n` +
      `${chalk.gray(`Запрос:       `)}${chalk.white(`${promptTokenCount} + ${candidatesTokenCount} = ${usageTokenCount} токенов`)}\n` +
      `${chalk.gray(`За день:      `)}${chalk.white(`${totalTokens} токенов (${totalRequests} запросов)`)}\n` +
      `${chalk.yellow.bold(`Стоимость:    ~$${estimatedCost.toFixed(4)}`)}`,
      { padding: 1, margin: { top: 1 }, borderStyle: 'round', borderColor: 'cyan' }
    );
    console.log(statsBox);
  } catch (error) {
    spinner.fail(chalk.red('Ошибка'));
    console.error(chalk.red("❌:"), formatModelError(error));
  }
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
  isQuotaError,
  sanitizeFilePath
};
