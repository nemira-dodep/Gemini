#!/usr/bin/env node

const { GoogleGenerativeAI } = require("@google/generative-ai");
const fs = require("fs").promises;
const path = require("path");
const chalk = require("chalk");
const ora = require("ora");
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
const STATS_FILE = path.join(__dirname, 'gemini-usage.json');

const MODELS_CONFIG = {
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

const systemInstruction = `Ты — экспертный CLI-помощник.

ВАЖНО: Создавай файлы ТОЛЬКО если пользователь явно просит! Например:
- "напиши функцию" → просто выведи код
- "создай файл config.js" → используй формат <file>
- "сделай index.html" → используй формат <file>

Если нужно создать файл, используй формат:
<file path="путь/к/файлу.расширение">содержимое файла</file>

Не придумывай сохранение файлов без явного запроса!`;

// --- ЛОГИКА ---

async function updateAndGetUsage(modelKey, tokenCount = 0) {
  let stats = { lastReset: new Date().toDateString(), tokens: {}, requests: {} };
  try {
    const data = await fs.readFile(STATS_FILE, 'utf8');
    stats = JSON.parse(data);
  } catch (e) {}
  
  // Сброс статистики при смене дня
  if (stats.lastReset !== new Date().toDateString()) {
    stats.tokens = {};
    stats.requests = {};
    stats.lastReset = new Date().toDateString();
  }
  
  // Подсчет токенов и запросов
  stats.tokens[modelKey] = (stats.tokens[modelKey] || 0) + tokenCount;
  stats.requests[modelKey] = (stats.requests[modelKey] || 0) + 1;
  
  await fs.writeFile(STATS_FILE, JSON.stringify(stats, null, 2));
  return stats;
}

async function scanModels() {
  const spinner = ora(chalk.cyan('Запрашиваю доступные модели у Google...')).start();
  try {
    const key = getApiKey();
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

    console.log(chalk.gray('\nСкопируйте нужный ID в MODELS_CONFIG вашего скрипта.'));
  } catch (error) {
    spinner.fail(chalk.red('Не удалось получить список моделей.'));
    console.error(chalk.red("❌ Детали:"), error.message);
    console.log(chalk.gray("\nПопробуйте обновить библиотеку: npm install @google/generative-ai@latest"));
  }
  setTimeout(() => process.exit(0), 100);
}

function getApiKey() {
  const key = process.env.GEMINI_API_KEY;
  if (!key) {
    console.error(chalk.red("❌ Ошибка: GEMINI_API_KEY не найден."));
    process.exit(1);
  }
  return key;
}

const genAI = new GoogleGenerativeAI(getApiKey());

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

async function showStats(modelKey = null) {
  try {
    const data = await fs.readFile(STATS_FILE, 'utf8');
    let stats = JSON.parse(data);
    
    // Конвертирование старого формата в новый
    if (stats.counts && !stats.tokens) {
      stats = { lastReset: stats.lastReset, tokens: {}, requests: {} };
    }
    stats.tokens = stats.tokens || {};
    stats.requests = stats.requests || {};
    
    console.log(chalk.magenta.bold(`\n📊 СТАТИСТИКА ИСПОЛЬЗОВАНИЯ`));
    console.log(chalk.gray(`Дата: ${stats.lastReset}\n`));

    if (modelKey && stats.tokens[modelKey] !== undefined) {
      // Статистика по одной модели
      const tokens = stats.tokens[modelKey] || 0;
      const requests = stats.requests[modelKey] || 0;
      const tokenPrice = modelKey === "gemma" ? 0.03 : (modelKey === "lite" ? 0.075 : 0.15);
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
        
        const tokenPrice = model === "gemma" ? 0.03 : (model === "lite" ? 0.075 : 0.15);
        const cost = (tokens / 1000) * tokenPrice;
        
        totalTokens += tokens;
        totalRequests += requests;
        totalCost += cost;

        console.log(chalk.cyan(`${model.padEnd(10)} │ ${requests.toString().padEnd(3)} запросов │ ${tokens.toString().padEnd(6)} токенов │ $${cost.toFixed(4)}`));
      }

      if (!hasData) {
        console.log(chalk.yellow("Нет данных. Сделайте первый запрос с помощью: node cli.js"));
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

function showHelp() {
  console.log(chalk.magenta.bold("\n🤖 GEMINI CLI HELPER — ПОЛНАЯ СПРАВКА\n"));
  console.log(chalk.cyan.bold("Команды:"));
  console.log(`  gemini "запрос"                ${chalk.gray("- Обычный чат")}`);
  console.log(`  gemini "запрос" --save         ${chalk.gray("- Запрос + сохранение файлов")}`);
  console.log(`  gemini --scan                  ${chalk.gray("- Проверить доступные модели Google")}`);
  console.log(`  gemini --stats                 ${chalk.gray("- Показать статистику за день")}`);
  console.log(`  gemini --stats <model>         ${chalk.gray("- Статистика по одной модели")}`);
  console.log(chalk.cyan.bold("Флаги:"));
  console.log(`  -c, --config       ${chalk.gray("- Выбор модели через меню стрелками")}`);
  console.log(`  -m, --model <key>  ${chalk.gray("- Указать модель (flash, lite, gemma, research, vision)")}`);
  console.log(`  --save             ${chalk.gray("- Создавать файлы из ответов")}`);
  console.log(`  -d, --dry-run      ${chalk.gray("- Показать файлы без сохранения")}`);
  console.log(`  -h, --help         ${chalk.gray("- Показать эту справку")}`);
}

async function run() {
  const args = process.argv.slice(2);
  let dryRun = false, saveFiles = false, selectedModelKey = "flash";
  const positional = [];

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--scan') { await scanModels(); return; }
    if (arg === '--stats') {
      const modelKey = args[i+1] && !args[i+1].startsWith('-') ? args[i+1] : null;
      await showStats(modelKey);
      return;
    }
    if (arg === '-h' || arg === '--help') { showHelp(); process.exit(0); }

    if (arg === '-c' || arg === '--config') {
      const { modelKey } = await inquirer.prompt([{
        type: 'list', name: 'modelKey', message: 'Выберите модель:',
        choices: Object.keys(MODELS_CONFIG).map(k => ({
          name: `${chalk.bold(k.padEnd(8))} | ${MODELS_CONFIG[k].desc}`, value: k
        }))
      }]);
      selectedModelKey = modelKey;
      continue;
    }
    if ((arg === '-m' || arg === '--model') && args[i+1]) { selectedModelKey = args[i+1]; i++; continue; }
    if (arg === '--save') saveFiles = true;
    else if (arg === '-d' || arg === '--dry-run') dryRun = true;
    else positional.push(arg);
  }

  const stdinContent = await readStdin();
  const inputSource = positional.length > 0 ? positional : (stdinContent ? [stdinContent] : []);
  if (inputSource.length === 0) { showHelp(); process.exit(1); }

  let prompt = "";
  const filesToProcess = [];
  let promptStarted = false;

  for (const arg of inputSource) {
    if (promptStarted) { prompt += " " + arg; continue; }
    try {
      const stats = await fs.stat(arg);
      if (stats.isFile()) filesToProcess.push(arg);
      else { promptStarted = true; prompt += arg; }
    } catch { promptStarted = true; prompt += arg; }
  }

  const modelCfg = MODELS_CONFIG[selectedModelKey] || MODELS_CONFIG["flash"];
  const model = genAI.getGenerativeModel({
    model: modelCfg.id,
    systemInstruction,
    tools: [{ google_search: {} }]
  });

  const spinner = ora({ text: chalk.magenta(`Gemini [${selectedModelKey}] на связи...`), color: 'magenta' }).start();

  try {
    const fileParts = [];
    for (const f of filesToProcess) {
      const data = await fs.readFile(f);
      fileParts.push({ inlineData: { data: data.toString("base64"), mimeType: "text/plain" }});
    }

    const result = await model.generateContent([prompt, ...fileParts]);
    const responseText = result.response.text();
    const usage = result.response.usageMetadata;
    
    spinner.succeed(chalk.green('Готово!\n'));
    console.log(marked.parse(responseText));

    const stats = await updateAndGetUsage(selectedModelKey, usage?.totalTokenCount || 0);
    const totalTokens = stats.tokens[selectedModelKey] || 0;
    const totalRequests = stats.requests[selectedModelKey] || 0;
    
    // Примерный расчет (flash обычно дороже)
    const tokenPricePer1k = selectedModelKey === "gemma" ? 0.03 : (selectedModelKey === "lite" ? 0.075 : 0.15);
    const estimatedCost = (totalTokens / 1000) * tokenPricePer1k;

    console.log(chalk.gray(`─`.repeat(50)));
    console.log(chalk.cyan(`📊 СТАТИСТИКА [${selectedModelKey}]:`));
    console.log(chalk.gray(`   Запрос:       ${usage?.promptTokenCount || '?'} входных + ${usage?.candidatesTokenCount || '?'} выходных = ${usage?.totalTokenCount || '?'} токенов`));
    console.log(chalk.gray(`   За день:      ${totalTokens} токенов (${totalRequests} запросов)`));
    console.log(chalk.gray(`   Примерная стоимость: ~$${estimatedCost.toFixed(4)}`));
    console.log(chalk.gray(`─`.repeat(50) + '\n'));

    // Обработка файлов только если явно указан флаг --save
    if (saveFiles) {
      const fileRegex = /<file path="([^"]+)">([\s\S]*?)<\/file>/g;
      let match;
      let foundFiles = false;
      while ((match = fileRegex.exec(responseText)) !== null) {
        foundFiles = true;
        const rawPath = match[1], fileContent = match[2].trim(), safePath = sanitizeFilePath(rawPath);
        if (!safePath) continue;
        if (dryRun) { console.log(chalk.cyan(`[dry-run] Сохранил бы: ${rawPath}`)); continue; }
        try {
          await fs.mkdir(path.dirname(safePath), { recursive: true });
          await fs.writeFile(safePath, fileContent, 'utf8');
          console.log(chalk.green(`💾 Сохранен: `) + chalk.bold(rawPath));
        } catch (err) {
          console.error(chalk.red(`❌ Ошибка сохранения ${rawPath}:`), err.message);
        }
      }
      if (!foundFiles) {
        console.log(chalk.yellow(`ℹ️  В ответе нет блоков <file> для сохранения`));
      }
    } else {
      // Показать уведомление если в ответе есть файлы и --save не указан
      if (responseText.includes('<file path=')) {
        console.log(chalk.yellow(`ℹ️  Ответ содержит файлы. Используйте --save для сохранения.`));
      }
    }
  } catch (error) {
    spinner.fail(chalk.red('Ошибка'));
    console.error(chalk.red("❌:"), error.message);
  }
}

run();