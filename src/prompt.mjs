import { ChatPromptTemplate } from "@langchain/core/prompts";

import { TEMPLATE_INTRO } from './constants.mjs';
export function getPrompt(applyContext = false, promptIntro = TEMPLATE_INTRO) {
    const context = applyContext ?
`
<context>
{context}
</context>
` : '';

    const prompt =
      ChatPromptTemplate.fromTemplate(`${promptIntro}
${context}
Question: {input}`);

    return prompt;
}