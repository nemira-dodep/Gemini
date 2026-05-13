#!/usr/bin/env node

const { program } = require("commander");
const { GoogleGenerativeAI, SchemaType } = require("@google/generative-ai");
const fs = require("fs").promises;
const path = require("path");
const { glob } = require("glob");
const chalk = require("chalk");
const ora = require("ora");
const boxen = require("boxen");
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
const STATS_FILE = path.join(__dirname, 'gemini-usage.json');
const CONFIG_FILE = path.join(__dirname, 'gemini-config.json');

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

const systemInstructionBase = `Ты — экспертный CLI-помощник.`;

// --- ЛОГИКА ---

async function processSavedFiles(filesArray, isDryRun) {
  if (!filesArray || filesArray.length === 0) {
    console.log(chalk.yellow(`ℹ️  Файлы для сохранения не предложены.`));
    return;
  }
  for (const file of filesArray) {
    const rawPath = file.path;
    const fileContent = file.content;
    const safePath = sanitizeFilePath(rawPath);
    if (!safePath) continue;
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
  try {
    const data = await fs.readFile(CONFIG_FILE, 'utf8');
    return JSON.parse(data).defaultModel || "flash";
  } catch (e) {
    return "flash";
  }
}

async function setDefaultModel(modelKey) {
  let config = {};
  try {
    const data = await fs.readFile(CONFIG_FILE, 'utf8');
    config = JSON.parse(data);
  } catch (e) {}
  config.defaultModel = modelKey;
  await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));
}

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

    console.log(chalk.gray('\nСкопируйте нужный ID в MODELS_CONFIG вашего скрипта.'));
  } catch (error) {
    spinner.fail(chalk.red('Не удалось получить список моделей.'));
    console.error(chalk.red("❌ Детали:"), error.message);
    console.log(chalk.gray("\nПопробуйте обновить библиотеку: npm install @google/generative-ai@latest"));
  }
  setTimeout(() => process.exit(0), 100);
}

async function getApiKey() {
  if (process.env.GEMINI_API_KEY) return process.env.GEMINI_API_KEY;
  try {
    const data = await fs.readFile(CONFIG_FILE, 'utf8');
    const config = JSON.parse(data);
    if (config.apiKey) return config.apiKey;
  } catch (e) {}

  console.error(chalk.red("\n❌ Ошибка: API ключ не найден."));
  console.log(chalk.gray(`Пожалуйста, установите ключ с помощью команды: ${chalk.cyan('gemini --set-key "ваш_ключ"')}\n`));
  process.exit(1);
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
    .option('--stats [model]', 'Показать статистику использования')
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
  ${chalk.green("gemini")} ${chalk.yellow("-i")}
`);

  program.parse(process.argv);
  const options = program.opts();
  const positional = program.args;

  let dryRun = !!options.dryRun;
  let saveFiles = !!options.save;
  let interactiveMode = !!options.interactive;
  let selectedModelKey = await getDefaultModel();

  if (options.setKey) {
    let config = {};
    try { config = JSON.parse(await fs.readFile(CONFIG_FILE, 'utf8')); } catch (e) {}
    config.apiKey = options.setKey;
    await fs.writeFile(CONFIG_FILE, JSON.stringify(config, null, 2));
    console.log(chalk.green(`✅ API ключ успешно сохранён!`));
    if (positional.length === 0 && !interactiveMode) return;
  }

  if (options.scan) { await scanModels(); return; }
  if (options.stats !== undefined) {
    await showStats(options.stats === true ? null : options.stats);
    return;
  }

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

  if (options.model) { selectedModelKey = options.model; }

  const stdinContent = await readStdin();
  const inputSource = positional.length > 0 ? positional : (stdinContent ? [stdinContent] : []);
  if (inputSource.length === 0 && !interactiveMode) { 
    program.help(); 
  }

  let prompt = "";
  const filesToProcess = [];
  let promptStarted = false;

  for (const arg of inputSource) {
    if (promptStarted) { prompt += " " + arg; continue; }
    
    try {
      // Пытаемся раскрыть как глоб (например *.js)
      const matchedFiles = await glob(arg, { nodir: true, windowsPathsNoEscape: true });
      
      if (matchedFiles.length > 0) {
        filesToProcess.push(...matchedFiles);
      } else {
        // Если глоб ничего не нашел, проверяем прямое наличие файла
        try {
          const stats = await fs.stat(arg);
          if (stats.isFile()) filesToProcess.push(arg);
          else { promptStarted = true; prompt += arg; }
        } catch {
          // Если и файла нет, это часть промпта
          promptStarted = true;
          prompt += arg;
        }
      }
    } catch (e) {
      promptStarted = true;
      prompt += arg;
    }
  }

  const apiKey = await getApiKey();
  const genAI = new GoogleGenerativeAI(apiKey);

  const modelCfg = MODELS_CONFIG[selectedModelKey] || MODELS_CONFIG["flash"];
  
  const generationConfig = { maxOutputTokens: 8192 };
  let currentSystemInstruction = systemInstructionBase;

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
    currentSystemInstruction = `${systemInstructionBase}\nВАЖНО: Всегда возвращай валидный JSON. Твой текстовый ответ помести в поле "text". Если пользователь просит создать файлы, добавь их в массив "files". Если файлы не нужны, массив "files" должен быть пустым.`;
  } else {
    currentSystemInstruction = `${systemInstructionBase}\nВыводи текст как обычно (Markdown). Файлы сохранять не требуется.`;
  }

  const model = genAI.getGenerativeModel({
    model: modelCfg.id,
    systemInstruction: currentSystemInstruction,
    generationConfig,
    tools: [{ google_search: {} }]
  });

  if (interactiveMode) {
    const chat = model.startChat({ history: [] });

    console.log(chalk.magenta.bold('\n💬 Режим интерактивного чата включен (введите "exit" для выхода)'));
    console.log(chalk.gray(`Модель: ${selectedModelKey}\n`));

    if (prompt || filesToProcess.length > 0) {
      console.log(chalk.bgBlue.black.bold(" ВЫ ") + " " + (prompt || chalk.gray("(только файлы)")));
      const spinner = ora({ text: chalk.magenta('Ожидание ответа...'), color: 'magenta' }).start();
      try {
        const fileParts = [];
        for (const f of filesToProcess) {
          const data = await fs.readFile(f);
          fileParts.push({ inlineData: { data: data.toString("base64"), mimeType: getMimeType(f) }});
        }
        
        const messageParts = filesToProcess.length > 0 ? [prompt, ...fileParts] : prompt;
        let responseText = "";
        let usageTokenCount = 0;
        if (saveFiles || dryRun) {
          const result = await chat.sendMessage(messageParts);
          spinner.succeed(chalk.green('Ответ получен:\n'));
          responseText = result.response.text();
          usageTokenCount = result.response.usageMetadata?.totalTokenCount || 0;
          try {
            const parsed = JSON.parse(responseText);
            console.log(boxen(marked(parsed.text), { padding: 1, borderColor: 'magenta', borderStyle: 'round' }));
            await processSavedFiles(parsed.files, dryRun);
          } catch(e) { console.log(marked(responseText) + "\n"); }
        } else {
          const result = await chat.sendMessageStream(messageParts);
          spinner.succeed(chalk.green('Ответ:\n'));
          process.stdout.write(chalk.bgMagenta.black.bold(" GEMINI ") + " ");
          for await (const chunk of result.stream) {
            const chunkText = chunk.text();
            responseText += chunkText;
            process.stdout.write(chunkText);
          }
          console.log("\n");
          const response = await result.response;
          usageTokenCount = response.usageMetadata?.totalTokenCount || 0;
        }
        await updateAndGetUsage(selectedModelKey, usageTokenCount);
      } catch (e) {
        spinner.fail(chalk.red('Ошибка'));
        console.error(chalk.red("❌:"), e.message);
      }
    }

    while (true) {
      const { userPrompt } = await inquirer.prompt([{
        type: 'input',
        name: 'userPrompt',
        prefix: '',
        message: chalk.bgBlue.black.bold(" ВЫ "),
      }]);

      if (!userPrompt || userPrompt.trim() === '') continue;
      if (userPrompt.trim().toLowerCase() === 'exit' || userPrompt.trim().toLowerCase() === 'quit') {
        console.log(chalk.gray('Завершение чата.'));
        break;
      }

      const spinner = ora({ text: chalk.magenta('Ожидание ответа...'), color: 'magenta' }).start();
      try {
        let responseText = "";
        let usageTokenCount = 0;
        if (saveFiles || dryRun) {
          const result = await chat.sendMessage(userPrompt);
          spinner.succeed(chalk.green('Ответ получен:\n'));
          responseText = result.response.text();
          usageTokenCount = result.response.usageMetadata?.totalTokenCount || 0;
          try {
            const parsed = JSON.parse(responseText);
            console.log(boxen(marked(parsed.text), { padding: 1, borderColor: 'magenta', borderStyle: 'round' }));
            await processSavedFiles(parsed.files, dryRun);
          } catch(e) { console.log(marked(responseText) + "\n"); }
        } else {
          const result = await chat.sendMessageStream(userPrompt);
          spinner.succeed(chalk.green('Ответ:\n'));
          process.stdout.write(chalk.bgMagenta.black.bold(" GEMINI ") + " ");
          for await (const chunk of result.stream) {
            const chunkText = chunk.text();
            responseText += chunkText;
            process.stdout.write(chunkText);
          }
          console.log("\n");
          const response = await result.response;
          usageTokenCount = response.usageMetadata?.totalTokenCount || 0;
        }
        await updateAndGetUsage(selectedModelKey, usageTokenCount);
      } catch (e) {
        spinner.fail(chalk.red('Ошибка'));
        console.error(chalk.red("❌:"), e.message);
      }
    }
    return;
  }

  const spinner = ora({ text: chalk.magenta(`Gemini [${selectedModelKey}] на связи...`), color: 'magenta' }).start();

  try {
    const fileParts = [];
    for (const f of filesToProcess) {
      const data = await fs.readFile(f);
      const mimeType = getMimeType(f);
      fileParts.push({ inlineData: { data: data.toString("base64"), mimeType }});
    }

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
    
    const tokenPricePer1k = selectedModelKey === "gemma" ? 0.03 : (selectedModelKey === "lite" ? 0.075 : 0.15);
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
    console.error(chalk.red("❌:"), error.message);
  }
}

run();