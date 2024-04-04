import { fileURLToPath } from "url";
import path from "path";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { MarkdownTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";

import { MODEL_DIRECTORY, LLAMA_MODEL } from './constants.mjs';
import Log from './logger.mjs';
import { get } from 'http';
const log = new Log('node-rag.log');

export async function getLlamaModel(rootPath = process.cwd(), modelDirectory = MODEL_DIRECTORY, llamaModel = LLAMA_MODEL) {
    await log
        .toConsole('Loading model', false, true)
        .infoToFileAsync();

    const modelPath = path.join(rootPath, modelDirectory, llamaModel);
    const { LlamaCpp } = await import("@langchain/community/llms/llama_cpp");
    const model = await new LlamaCpp({
        modelPath: modelPath,
        batchSize: 1024,
        temperature: 0.9,
        gpuLayers: 64
    });

    return model;
}

export async function getOpenAIModel() {
    await log
        .toConsole('Loading model', false, true)
        .infoToFileAsync();

    const { ChatOpenAI } = await import("@langchain/openai");
    const key = await import('../key.json', { with: { type: 'json' } });
    const model = new ChatOpenAI({
        temperature: 0.9,
        openAIApiKey: key.default.apiKey
    });

    return model;
}

export async function getOpenShiftAiModel() {
    const { ChatOpenAI } = await import("@langchain/openai");
    model = new ChatOpenAI(
        {
            temperature: 0.9,
            openAIApiKey: 'EMPTY',
            modelName: 'mistralai/Mistral-7B-Instruct-v0.2'
        },
        { baseURL: 'http://vllm.llm-hosting.svc.cluster.local:8000/v1' }
    );

    return model;
}

/**
 * @param  {'llama-cpp'|'openAI'|'Openshift.ai'} modelType
 * @returns {Promise<any>} either a llama model, an openAI model, or an Openshift.ai model
 */
export async function getModel(modelType) {
    await log
        .toConsole('Loading model', false, true)
        .infoToFileAsync();

    let model;
    if (modelType === 'llama-cpp') {
        model = await getLlamaModel();
    } else if (modelType === 'openAI') {
        model = await getOpenAIModel();
    } else if (modelType === 'Openshift.ai') {
        model = await getOpenShiftAiModel();
        setInterval(() => {
            console.log('keep-alive');
        }, 5000);
    };

    return model;
};

/** 
 * @description gets data from a directory and processes it into a retriever
 * @param  {string} dataDirectory
 * @returns {Promise<Retriever>|null}
 */
export async function getAugmentedData(dataDirectory) {
  let retriever = null;

  try {
    if (dataDirectory) {
      await log
      .toConsole(`Loading and processing augmenting data from ${dataDirectory}`, true, true)
      .infoToFileAsync();
      
      const docLoader = new DirectoryLoader(
        dataDirectory,
        {
          ".md": (path) => new TextLoader(path),
        }
      );
      const docs = await docLoader.load();
      
      const splitter = await new MarkdownTextSplitter({
        chunkSize: 500,
        chunkOverlap: 50
      });
      const splitDocs = await splitter.splitDocuments(docs);
      
      const vectorStore = await MemoryVectorStore.fromDocuments(
        splitDocs,
        new HuggingFaceTransformersEmbeddings()
      );
        
      retriever = await vectorStore.asRetriever();
        
      await log
        .toConsole("Augmenting data loaded", false, true)
        .infoToFileAsync();
    }
  } catch (getAugmentedDataError) {
    await log
    .toConsole("Error loading augmenting data", true, true)
    .errorToFileAsync(getAugmentedDataError);
  }

  return retriever;
}

/**
 * @description creates a chain with context
 * @param  {Object} baseModel a model created from getModel
 * @param  {Object} prompt a prompt created from getPrompt
 * @param  {string} contextDirectory
 * @returns {Promise<Chain>}
 */
export async function getChainWithContext(baseModel, prompt, contextDirectory) {
    if (!baseModel || !prompt || !contextDirectory) {
        throw new Error('Base model, prompt, and context directory required');
    }
    const documentChain = await createStuffDocumentsChain({
        llm: baseModel,
        prompt,
    });
    const retriever = await getAugmentedData(contextDirectory);
    const retrievalChain = await createRetrievalChain({
        combineDocsChain: documentChain,
        retriever,
    });

    return retrievalChain;
}

/**
 * @param  {Object} prompt a prompt created from getPrompt
 * @param  {Object} baseModel a model created from getModel
 * @param  {string} [contextDirectory] a directory containing source documents
 */
export async function getChain(prompt, baseModel, contextDirectory) {
    if (!prompt || !baseModel) {
        throw new Error('Prompt and base model required');
    }
  let model = prompt.pipe(baseModel);
  const shouldApplyContext = !!contextDirectory;

  try {
    if (shouldApplyContext) {
      model = getChainWithContext(baseModel, prompt, contextDirectory);
    }
  } catch (getChainError) {
    await log
    .toConsole("Error creating chain", true)
    .errorToFileAsync(getChainError);
  }
  
  return model;
}