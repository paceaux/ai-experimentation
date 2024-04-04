import {fileURLToPath} from "url";
import path from "path";



import yargs from 'yargs/yargs';
import { hideBin } from 'yargs/helpers';

import {
  MODEL_DIRECTORY,
  LLAMA_MODEL,
  OPENSHIFT_MODEL,
  DEFAULT_TEMPERATURE,
  DEFAULT_MODEL_TYPE
} from './src/constants.mjs';

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
  .option('contextDirectory', {
    alias: 'd',
    type: 'string',
    description: 'The directory containing the source documents',
  })
  .option('modelType', {
    alias: 'm',
    type: 'string',
    description: 'The type of model to use',
    'default': DEFAULT_MODEL_TYPE
  })
  .option('temperature', {
    alias: 't',
    type: 'number',
    description: 'The directory containing the source documents',
    default: DEFAULT_TEMPERATURE,
  })
  .help()
  .alias('help', 'h');

const {
  question,
  contextDirectory,
  temperature,
  modelType
} = argv;

const hasContext = !!contextDirectory;

const prompt = getPrompt(hasContext);
const model = await getModel(modelType, temperature);
const chain = await getChain(prompt, model, contextDirectory);

let result = await chain.invoke({
  input: question,
});

const resultMessage = result?.answer || result;

await log
  .toConsole(resultMessage, true)
  .infoToFileAsync(result);
