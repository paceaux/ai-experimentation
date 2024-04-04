# Summary

## Setup

1. `npm install`
2. Create a directory called `models` and copy the model to use into that directory.
  
    * Models
  can be downloaded from hugging face, for example - 
  [TheBloke/Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
3. Modify  `src/constants.mjs` as needed to use the default model type or name you want.
4. (optional) Create a directory called SOURCE_FILES and copy markdown documents into it with.

## Usage

like a CLI

Ask a basic question:

```bash
node node-rag.mjs -q <your question>
```

Ask a question with a context:

```bash
node node-rag.mjs -q <your question> -d <Path to Markdown>
```

### CLI options

| Option | Alias | Description |
| --- | --- | --- |
| `--question` | `-q` | String. Question to ask |
| `--contextDirectory` | `-d` | (optional) String. Directory containing source documents. |
| `--modelType` | `-m` | (optional) String. Default: 'llama-cpp'.  Options: 'llama-cpp', 'openAI', 'Openshift.ai'  |
| `--temperature` | `-t` | (Optional) Number. Default: 0.9. Temperature to use. |
