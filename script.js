#!/usr/bin/env node

const { GoogleGenerativeAI } = require("@google/generative-ai");
const fs = require("fs").promises;
const path = require("path");
const chalk = require("chalk");
const ora = require("ora");
const { marked } = require("marked");
const { markedTerminal } = require("marked-terminal");

// Поддержка .env файла (опционально, если установлен dotenv)
try { require("dotenv").config(); } catch {}

marked.use(markedTerminal({
  code: chalk.green,
  firstHeading: chalk.blue.bold,
  heading: chalk.cyan.bold,
  text: chalk.white,
}));

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const RETRY_CODES = [429, 500, 502, 503];
const RETRIES = 3;
const STATS_FILE = path.join(__dirname, 'gemini-usage.json');

function getApiKey() {
  const key = process.env.GEMINI_API_KEY;
  if (!key) {
    console.error(chalk.red("❌ Ошибка: переменная окружения GEMINI_API_KEY не найдена."));
    console.error(chalk.gray("   Установите её в окружении или в файле .env"));
    process.exit(1);
  }
  return key;
}

async function updateAndGetUsage(modelKey, tokenCount = 0) {
  let stats = { lastReset: new Date().toDateString(), tokens: {}, requests: {} };
  try {
    const data = await fs.readFile(STATS_FILE, 'utf8');
    stats = JSON.parse(data);
  } catch (e) {}
  
  if (stats.lastReset !== new Date().toDateString()) {
    stats.tokens = {};
    stats.requests = {};
    stats.lastReset = new Date().toDateString();
  }
  
  stats.tokens[modelKey] = (stats.tokens[modelKey] || 0) + tokenCount;
  stats.requests[modelKey] = (stats.requests[modelKey] || 0) + 1;
  
  await fs.writeFile(STATS_FILE, JSON.stringify(stats, null, 2));
  return stats;
}

const genAI = new GoogleGenerativeAI(getApiKey());
const modelName = process.env.GEMINI_MODEL || "gemini-2.5-flash";

const systemInstruction = `Ты — экспертный CLI-помощник.
Если нужно создать или изменить файл, используй формат:
<file path="относительный/путь/имя.расширение">
содержимое файла
</file>`;

async function main() {
  const prompt = process.argv.slice(2).join(" ");
  
  if (!prompt) {
    console.error(chalk.red("❌ Использование: node script.js \"ваш запрос\""));
    console.error(chalk.gray("Пример: node script.js \"Напиши функцию для сортировки массива\""));
    process.exit(1);
  }

  const model = genAI.getGenerativeModel({
    model: modelName,
    systemInstruction,
  });

  const spinner = ora({ text: chalk.magenta(`${modelName} обрабатывает запрос...`), color: 'magenta' }).start();

  try {
    const result = await model.generateContent(prompt);
    const responseText = result.response.text();
    const usage = result.response.usageMetadata;

    spinner.succeed(chalk.green('Готово!\n'));
    console.log(marked.parse(responseText));

    const stats = await updateAndGetUsage(modelName, usage?.totalTokenCount || 0);
    const totalTokens = stats.tokens[modelName] || 0;
    const totalRequests = stats.requests[modelName] || 0;
    const tokenPrice = 0.15;
    const estimatedCost = (totalTokens / 1000) * tokenPrice;

    console.log(chalk.gray(`─`.repeat(50)));
    console.log(chalk.cyan(`📊 СТАТИСТИКА:`));
    console.log(chalk.gray(`   Этот запрос: ${usage?.promptTokenCount || '?'} входных + ${usage?.candidatesTokenCount || '?'} выходных = ${usage?.totalTokenCount || '?'} токенов`));
    console.log(chalk.gray(`   За день:     ${totalTokens} токенов (${totalRequests} запросов)`));
    console.log(chalk.gray(`   Стоимость:   ~$${estimatedCost.toFixed(4)}`));
    console.log(chalk.gray(`─`.repeat(50)));
  } catch (error) {
    spinner.fail(chalk.red('Ошибка'));
    console.error(chalk.red("❌:"), error.message);
    process.exit(1);
  }
}

main();
