import {fileURLToPath} from "url";
import path from "path";



import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';
import Log from './src/logger.mjs';
const log = new Log('node-rag.log');

import { getModel, getAugmentedData, getChain } from './src/model.mjs';
import { getPrompt } from './src/prompt.mjs';

const { argv } = yargs(hideBin(process.argv))
  .option('question', {
    alias: 'q',
    type: 'string',
    description: 'The question to ask the model',
    'default': 'How do I build a good container for a Node.js application'
  })
  .option('directory', {
    alias: 'd',
    type: 'string',
    description: 'The directory containing the source documents',
  })
  .help()
  .alias('help', 'h');

const {
  question,
  directory,
} = argv;

const hasContext = !!directory;

const prompt = getPrompt(hasContext);
const model = await getModel('llama-cpp');
const chain = await getChain(prompt, model, directory);

let result = await chain.invoke({
  input: question,
});

const resultMessage = result?.answer || result;

await log
  .toConsole(resultMessage, true)
  .infoToFileAsync(result);
