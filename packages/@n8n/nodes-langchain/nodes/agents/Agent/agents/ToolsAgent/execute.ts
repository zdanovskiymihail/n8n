import { BaseCallbackHandler } from '@langchain/core/callbacks/base';
import type { LLMResult } from '@langchain/core/outputs';
import type { BaseMemory } from '@langchain/core/memory';
import type { BaseChatModel } from '@langchain/core/language_models/chat_models';
// Import specific message types
import { AIMessage, HumanMessage, ToolMessage, BaseMessage } from '@langchain/core/messages'; // Ensure BaseMessage is imported
import type { InvalidToolCall, ToolCall } from '@langchain/core/messages/tool';
import type { BaseMessagePromptTemplateLike } from '@langchain/core/prompts';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import type { Tool } from '@langchain/core/tools';
import { DynamicStructuredTool } from '@langchain/core/tools';
import type { AgentAction, AgentFinish } from 'langchain/agents';
import { AgentExecutor, createToolCallingAgent } from 'langchain/agents';
import type { ToolsAgentAction } from 'langchain/dist/agents/tool_calling/output_parser';
import { omit } from 'lodash';
import { BINARY_ENCODING, jsonParse, NodeConnectionTypes, NodeOperationError } from 'n8n-workflow';
import type { IExecuteFunctions, INodeExecutionData } from 'n8n-workflow'; // Removed Logger import
import type { ZodObject } from 'zod';
import { z } from 'zod';

import { isChatInstance, getPromptInputByType, getConnectedTools } from '@utils/helpers';
import { getOptionalOutputParser, type N8nOutputParser } from '@utils/output_parsers/N8nOutputParser';

import { SYSTEM_MESSAGE } from './prompt';

/* -----------------------------------------------------------
   Output Parser Helper
----------------------------------------------------------- */
/**
 * Retrieve the output parser schema.
 * If the parser does not return a valid schema, default to a schema with a single text field.
 */
export function getOutputParserSchema(
	outputParser: N8nOutputParser,
	// eslint-disable-next-line @typescript-eslint/no-explicit-any
): ZodObject<any, any, any, any> {
	// ...existing code...
	const schema = (outputParser.getSchema() as ZodObject<any, any, any, any>) ?? z.object({ text: z.string() });
	return schema;
}

/* -----------------------------------------------------------
   Binary Data Helpers
----------------------------------------------------------- */
/**
 * Extracts binary image messages from the input data.
 * When operating in filesystem mode, the binary stream is first converted to a buffer.
 *
 * @param ctx - The execution context
 * @param itemIndex - The current item index
 * @returns A HumanMessage containing the binary image messages.
 */
export async function extractBinaryMessages(
	ctx: IExecuteFunctions,
	itemIndex: number,
): Promise<HumanMessage> {
	const binaryData = ctx.getInputData()?.[itemIndex]?.binary ?? {};
	const binaryMessages = await Promise.all(
		Object.values(binaryData)
			.filter((data) => data.mimeType.startsWith('image/'))
			.map(async (data) => {
				let binaryUrlString: string;

				// In filesystem mode we need to get binary stream by id before converting it to buffer
				if (data.id) {
					const binaryBuffer = await ctx.helpers.binaryToBuffer(
						await ctx.helpers.getBinaryStream(data.id),
					);
					binaryUrlString = `data:${data.mimeType};base64,${Buffer.from(binaryBuffer).toString(
						BINARY_ENCODING,
					)}`;
				} else {
					binaryUrlString = data.data.includes('base64')
						? data.data
						: `data:${data.mimeType};base64,${data.data}`;
				}

				return {
					type: 'image_url',
					image_url: {
						url: binaryUrlString,
					},
				};
			}),
	);
	return new HumanMessage({
		content: [...binaryMessages],
	});
}

/* -----------------------------------------------------------
   Agent Output Format Helpers
----------------------------------------------------------- */
/**
 * Fixes empty content messages in agent steps.
 *
 * This function is necessary when using RunnableSequence.from in LangChain.
 * If a tool doesn't have any arguments, LangChain returns input: '' (empty string).
 * This can throw an error for some providers (like Anthropic) which expect the input to always be an object.
 * This function replaces empty string inputs with empty objects to prevent such errors.
 *
 * @param steps - The agent steps to fix
 * @returns The fixed agent steps
 */
export function fixEmptyContentMessage(
	steps: AgentFinish | ToolsAgentAction[],
): AgentFinish | ToolsAgentAction[] {
	if (!Array.isArray(steps)) return steps;

	steps.forEach((step) => {
		if ('messageLog' in step && step.messageLog !== undefined) {
			if (Array.isArray(step.messageLog)) {
				step.messageLog.forEach((message: BaseMessage) => {
					if ('content' in message && Array.isArray(message.content)) {
						// eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
						(message.content as Array<{ input?: string | object }>).forEach((content) => {
							if (content.input === '') {
								content.input = {};
							}
						});
					}
				});
			}
		}
	});

	return steps;
}

/**
 * Ensures consistent handling of outputs regardless of the model used,
 * providing a unified output format for further processing.
 *
 * This method is necessary to handle different output formats from various language models.
 * Specifically, it checks if the agent step is the final step (contains returnValues) and determines
 * if the output is a simple string (e.g., from OpenAI models) or an array of outputs (e.g., from Anthropic models).
 *
 * Examples:
 * 1. Anthropic model output:
 * ```json
 *    {
 *      "output": [
 *        {
 *          "index": 0,
 *          "type": "text",
 *          "text": "The result of the calculation is approximately 1001.8166..."
 *        }
 *      ]
 *    }
 *```
 * 2. OpenAI model output:
 * ```json
 *    {
 *      "output": "The result of the calculation is approximately 1001.82..."
 *    }
 * ```
 *
 * @param steps - The agent finish or agent action steps.
 * @returns The modified agent finish steps or the original steps.
 */
export function handleAgentFinishOutput(
	steps: AgentFinish | AgentAction[],
): AgentFinish | AgentAction[] {
	type AgentMultiOutputFinish = AgentFinish & {
		returnValues: { output: Array<{ text: string; type: string; index: number }> };
	};
	const agentFinishSteps = steps as AgentMultiOutputFinish | AgentFinish;

	if (agentFinishSteps.returnValues) {
		const isMultiOutput = Array.isArray(agentFinishSteps.returnValues?.output);
		if (isMultiOutput) {
			// If all items in the multi-output array are of type 'text', merge them into a single string
			const multiOutputSteps = agentFinishSteps.returnValues.output as Array<{
				index: number;
				type: string;
				text: string;
			}>;
			const isTextOnly = multiOutputSteps.every((output) => 'text' in output);
			if (isTextOnly) {
				agentFinishSteps.returnValues.output = multiOutputSteps
					.map((output) => output.text)
					.join('\n')
					.trim();
			}
			return agentFinishSteps;
		}
	}

	return agentFinishSteps;
}

/**
 * Wraps the parsed output so that it can be stored in memory.
 * If memory is connected, the output is stringified.
 *
 * @param output - The parsed output object
 * @param memory - The connected memory (if any)
 * @returns The formatted output object
 */
export function handleParsedStepOutput(
	output: Record<string, unknown>,
	memory?: BaseMemory,
): { returnValues: Record<string, unknown>; log: string } {
	return {
		returnValues: memory ? { output: JSON.stringify(output) } : output,
		log: 'Final response formatted',
	};
}

/**
 * Parses agent steps using the provided output parser.
 * If the agent used the 'format_final_json_response' tool, the output is parsed accordingly.
 *
 * @param steps - The agent finish or action steps
 * @param outputParser - The output parser (if defined)
 * @param memory - The connected memory (if any)
 * @returns The parsed steps with the final output
 */
export const getAgentStepsParser =
	(outputParser?: N8nOutputParser, memory?: BaseMemory) =>
	async (steps: AgentFinish | AgentAction[]): Promise<AgentFinish | AgentAction[]> => {
		// Check if the steps contain the 'format_final_json_response' tool invocation.
		if (Array.isArray(steps)) {
			const responseParserTool = steps.find((step) => step.tool === 'format_final_json_response');
			if (responseParserTool && outputParser) {
				const toolInput = responseParserTool.toolInput;
				// Ensure the tool input is a string
				const parserInput = toolInput instanceof Object ? JSON.stringify(toolInput) : toolInput;
				const returnValues = (await outputParser.parse(parserInput)) as Record<string, unknown>;
				return handleParsedStepOutput(returnValues, memory);
			}
		}

		// Otherwise, if the steps contain a returnValues field, try to parse them manually.
		if (outputParser && typeof steps === 'object' && (steps as AgentFinish).returnValues) {
			const finalResponse = (steps as AgentFinish).returnValues;
			let parserInput: string;

			if (finalResponse instanceof Object) {
				if ('output' in finalResponse) {
					try {
						// If the output is an object, we will try to parse it as JSON
						// this is because parser expects stringified JSON object like { "output": { .... } }
						// so we try to parse the output before wrapping it and then stringify it
						parserInput = JSON.stringify({ output: jsonParse(finalResponse.output) });
					} catch (error) {
						// Fallback to the raw output if parsing fails.
						parserInput = finalResponse.output;
					}
				} else {
					// If the output is not an object, we will stringify it as it is
					parserInput = JSON.stringify(finalResponse);
				}
			} else {
				parserInput = finalResponse;
			}

			const returnValues = (await outputParser.parse(parserInput)) as Record<string, unknown>;
			return handleParsedStepOutput(returnValues, memory);
		}

		return handleAgentFinishOutput(steps);
	};

/* -----------------------------------------------------------
   Agent Setup Helpers
----------------------------------------------------------- */
/**
 * Retrieves the language model from the input connection.
 * Throws an error if the model is not a valid chat instance or does not support tools.
 *
 * @param ctx - The execution context
 * @returns The validated chat model
 */
export async function getChatModel(ctx: IExecuteFunctions): Promise<BaseChatModel> {
	// ...existing code...
	const model = await ctx.getInputConnectionData(NodeConnectionTypes.AiLanguageModel, 0);
	if (!isChatInstance(model) || !model.bindTools) {
		throw new NodeOperationError(
			ctx.getNode(),
			'Tools Agent requires Chat Model which supports Tools calling',
		);
	}
	return model;
}

/**
 * Retrieves the memory instance from the input connection.
 *
 * @param ctx - The execution context
 * @returns The connected memory (if any)
 */
export async function getOptionalMemory(
	ctx: IExecuteFunctions,
): Promise<BaseMemory | undefined> {
	// Cast only as BaseMemory
	const memory = (await ctx.getInputConnectionData(NodeConnectionTypes.AiMemory, 0)) as BaseMemory | undefined;
	return memory;
}

/**
 * Retrieves the connected tools and (if an output parser is defined)
 * appends a structured output parser tool.
 *
 * @param ctx - The execution context
 * @param outputParser - The optional output parser
 * @returns The array of connected tools
 */
export async function getTools(
	ctx: IExecuteFunctions,
	outputParser?: N8nOutputParser,
): Promise<Array<DynamicStructuredTool | Tool>> {
	// ...existing code...
	const tools = (await getConnectedTools(ctx, true, false)) as Array<DynamicStructuredTool | Tool>;
	if (outputParser) {
		const schema = getOutputParserSchema(outputParser);
		const structuredOutputParserTool = new DynamicStructuredTool({
			schema,
			name: 'format_final_json_response',
			description:
				'Use this tool to format your final response to the user in a structured JSON format. This tool validates your output against a schema to ensure it meets the required format. ONLY use this tool when you have completed all necessary reasoning and are ready to provide your final answer. Do not use this tool for intermediate steps or for asking questions. The output from this tool will be directly returned to the user.',
			func: async () => '',
		});
		tools.push(structuredOutputParserTool);
	}
	return tools;
}

/**
 * Prepares the prompt messages for the agent.
 *
 * @param ctx - The execution context
 * @param itemIndex - The current item index
 * @param options - Options containing systemMessage and other parameters
 * @returns The array of prompt messages
 */
export async function prepareMessages(
	ctx: IExecuteFunctions,
	itemIndex: number,
	options: {
		systemMessage?: string;
		passthroughBinaryImages?: boolean;
		outputParser?: N8nOutputParser;
		context?: string; // Ensure context is part of the options type
	},
): Promise<BaseMessagePromptTemplateLike[]> {
	const useSystemMessage = options.systemMessage ?? ctx.getNode().typeVersion < 1.9;

	const messages: BaseMessagePromptTemplateLike[] = [];

	if (useSystemMessage) {
		messages.push([
			'system',
			`{system_message}${options.outputParser ? '\n\n{formatting_instructions}' : ''}`,
		]);
	} else if (options.outputParser) {
		messages.push(['system', '{formatting_instructions}']);
	}

	messages.push(['placeholder', '{chat_history}']);

	// Prepare the main input, potentially prepending context
	let mainInput = '{input}';
	if (options.context && options.context.trim() !== '') {
		mainInput = `<context>${options.context.trim()}</context>user_prompt:{input}`;
	}
	messages.push(['human', mainInput]);

	// If there is binary data and the node option permits it, add a binary message
	const hasBinaryData = ctx.getInputData()?.[itemIndex]?.binary !== undefined;
	if (hasBinaryData && options.passthroughBinaryImages) {
		const binaryMessage = await extractBinaryMessages(ctx, itemIndex);
		if (binaryMessage.content.length !== 0) {
			messages.push(binaryMessage);
		} else {
			ctx.logger.debug('Not attaching binary message, since its content was empty');
		}
	}

	// We add the agent scratchpad last, so that the agent will not run in loops
	// by adding binary messages between each interaction
	messages.push(['placeholder', '{agent_scratchpad}']);
	return messages;
}

/**
 * Creates the chat prompt from messages.
 *
 * @param messages - The messages array
 * @returns The ChatPromptTemplate instance
 */
export function preparePrompt(messages: BaseMessagePromptTemplateLike[]): ChatPromptTemplate {
	return ChatPromptTemplate.fromMessages(messages);
}

/* -----------------------------------------------------------
   Callback Handlers
----------------------------------------------------------- */

class MemoryStepsCallbackHandler extends BaseCallbackHandler {
	name = 'MemoryStepsCallbackHandler';
	private lastAction: ToolsAgentAction | undefined;
	private intermediateMessages: BaseMessage[] = [];

	constructor() {
		super();
	}

	async handleAgentAction(action: ToolsAgentAction): Promise<void> {
		this.lastAction = action;
	}

	async handleToolEnd(output: string, runId: string): Promise<void> {
		if (this.lastAction) {
			const action = this.lastAction;
			this.lastAction = undefined;
			const aiContent = `Thought: ${action.log}`;
			const toolCalls: ToolCall[] = [];
			if (action.toolCallId) {
				let args: Record<string, any> = {};
				try {
					args = typeof action.toolInput === 'string'
						? jsonParse(action.toolInput)
						: (action.toolInput as Record<string, any>);
				} catch {
					args = {};
				}
				// Use the object directly as required
				toolCalls.push({
					id: action.toolCallId,
					name: action.tool,
					args,
					type: 'tool_call',
				});
			}
			const aiMessage = new AIMessage({
				content: aiContent,
				tool_calls: toolCalls,
				invalid_tool_calls: [],
			});
			this.intermediateMessages.push(aiMessage);

			const toolMessage = new ToolMessage({
				content: output,
				tool_call_id: action.toolCallId ?? `tool_call_${runId}`,
				name: action.tool,
			});
			this.intermediateMessages.push(toolMessage);
		}
	}

	async handleToolError(error: Error | unknown, runId: string): Promise<void> {
		if (this.lastAction) {
			const action = this.lastAction;
			this.lastAction = undefined;
			const errorMessage = error instanceof Error ? error.message : String(error);
			const aiContent = `Thought: ${action.log}`;
			const toolCalls: ToolCall[] = [];
			const invalidToolCalls: InvalidToolCall[] = [];
			if (action.toolCallId) {
				let args: Record<string, any> = {};
				try {
					args = typeof action.toolInput === 'string'
						? jsonParse(action.toolInput)
						: (action.toolInput as Record<string, any>);
				} catch {
					args = {};
				}
				invalidToolCalls.push({
					id: action.toolCallId,
					name: action.tool,
					// We keep args as object to satisfy the type requirement
					args: JSON.stringify(args),
					error: errorMessage,
					type: 'invalid_tool_call',
				});
			}
			const aiMessage = new AIMessage({
				content: aiContent,
				tool_calls: toolCalls,
				invalid_tool_calls: invalidToolCalls,
			});
			this.intermediateMessages.push(aiMessage);

			const toolMessage = new ToolMessage({
				content: `Tool Error: ${errorMessage}`,
				tool_call_id: action.toolCallId ?? `tool_call_${runId}_error`,
				name: action.tool,
			});
			this.intermediateMessages.push(toolMessage);
		}
	}

	getIntermediateMessages(): BaseMessage[] {
		return this.intermediateMessages;
	}

	async handleAgentEnd(): Promise<void> {
		if (this.lastAction) {
			this.lastAction = undefined;
		}
	}
}

export class TokenUsageCallbackHandler extends BaseCallbackHandler {
	name = 'TokenUsageCallbackHandler';
	promptTokens = 0;
	completionTokens = 0;
	totalTokens = 0;

	async handleLLMEnd(output: LLMResult): Promise<void> {
		const tokenUsage = output.llmOutput?.tokenUsage;
		if (tokenUsage) {
			this.promptTokens += tokenUsage.promptTokens ?? 0;
			this.completionTokens += tokenUsage.completionTokens ?? 0;
			this.totalTokens += tokenUsage.totalTokens ?? 0;
		}
	}

	getUsage() {
		return {
			promptTokens: this.promptTokens,
			completionTokens: this.completionTokens,
			totalTokens: this.totalTokens,
		};
	}

	reset() {
		this.promptTokens = 0;
		this.completionTokens = 0;
		this.totalTokens = 0;
	}
}

/* -----------------------------------------------------------
   Main Executor Function
----------------------------------------------------------- */
/**
 * The main executor method for the Tools Agent.
 *
 * This function retrieves necessary components (model, memory, tools), prepares the prompt,
 * creates the agent, and processes each input item. The error handling for each item is also
 * managed here based on the node's continueOnFail setting.
 *
 * @returns The array of execution data for all processed items
 */
export async function toolsAgentExecute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
	const returnData: INodeExecutionData[] = [];
	const items = this.getInputData();
	const outputParser = await getOptionalOutputParser(this);
	const tools = await getTools(this, outputParser);
	for (let itemIndex = 0; itemIndex < items.length; itemIndex++) {
		const tokenCallbackHandler = new TokenUsageCallbackHandler();
		let memoryStepsCallbackHandler: MemoryStepsCallbackHandler | undefined;
		try {
			const model = await getChatModel(this);
			const memory = await getOptionalMemory(this);
			const rawInput = getPromptInputByType({
				ctx: this,
				i: itemIndex,
				inputKey: 'text',
				promptTypeKey: 'promptType',
			});
			if (rawInput === undefined) {
				throw new NodeOperationError(this.getNode(), 'The “text” parameter is empty.');
			}
			const options = this.getNodeParameter('options', itemIndex, {}) as {
				systemMessage?: string;
				maxIterations?: number;
				returnIntermediateSteps?: boolean;
				saveIntermediateStepsToMemory?: boolean;
				passthroughBinaryImages?: boolean;
				outputTokensConsumption?: boolean;
				context?: string;
			};
			const cleanedInput = rawInput
				.replace(/<context>[\s\S]*?<\/context>/g, '')
				.replace(/^ *user_prompt:/, '')
				.trim();
			const messages = await prepareMessages(this, itemIndex, {
				systemMessage: options.systemMessage,
				passthroughBinaryImages: options.passthroughBinaryImages ?? true,
				outputParser,
				context: options.context,
			});
			const prompt = preparePrompt(messages);
			const agent = createToolCallingAgent({
				llm: model,
				tools,
				prompt,
				streamRunnable: false,
			});
			agent.streamRunnable = false;
			const runnableAgent = RunnableSequence.from([
				agent,
				getAgentStepsParser(outputParser, memory),
				fixEmptyContentMessage,
			]);
			const shouldReturnIntermediateSteps =
				options.returnIntermediateSteps === true || options.saveIntermediateStepsToMemory === true;
			const callbacks: BaseCallbackHandler[] = [];
			if (options.outputTokensConsumption) {
				callbacks.push(tokenCallbackHandler);
			}
			if (memory && options.saveIntermediateStepsToMemory) {
				memoryStepsCallbackHandler = new MemoryStepsCallbackHandler();
				callbacks.push(memoryStepsCallbackHandler);
			}
			const executor = AgentExecutor.fromAgentAndTools({
				agent: runnableAgent,
				memory,
				tools,
				returnIntermediateSteps: shouldReturnIntermediateSteps,
				maxIterations: options.maxIterations ?? 10,
			});
			const response = await executor.invoke(
				{
					input: cleanedInput,
					system_message: options.systemMessage ?? SYSTEM_MESSAGE,
					formatting_instructions: outputParser
						? 'IMPORTANT: For your response to user, you MUST use the `format_final_json_response` tool with your complete answer formatted according to the required schema. Do not attempt to format the JSON manually - always use this tool. Your response will be rejected if it is not properly formatted через этот инструмент. Используйте этот инструмент только тогда, когда вы готовы предоставить свой окончательный ответ.'
						: undefined,
				},
				{
					signal: this.getExecutionCancelSignal(),
					callbacks,
				},
			);

			if (memory && options.saveIntermediateStepsToMemory && memoryStepsCallbackHandler) {
				const intermediateMessages = memoryStepsCallbackHandler.getIntermediateMessages();
				if (intermediateMessages.length > 0) {
					if (
						typeof (memory as any).chatHistory?.getMessages === 'function' &&
						typeof (memory as any).chatHistory?.clear === 'function' &&
						typeof (memory as any).chatHistory?.addMessages === 'function'
					) {
						try {
							const currentHistory: BaseMessage[] = await (memory as any).chatHistory.getMessages();
							let lastHumanIndex = -1;
							for (let i = currentHistory.length - 1; i >= 0; i--) {
								if (currentHistory[i]._getType() === 'human') {
									lastHumanIndex = i;
									break;
								}
							}
							let reorderedHistory: BaseMessage[] = [];
							if (lastHumanIndex !== -1) {
								reorderedHistory = [
									...currentHistory.slice(0, lastHumanIndex + 1),
									...intermediateMessages,
									...currentHistory.slice(lastHumanIndex + 1),
								];
							} else {
								reorderedHistory = [...currentHistory, ...intermediateMessages];
							}
							await (memory as any).chatHistory.clear();
							await (memory as any).chatHistory.addMessages(reorderedHistory);
						} catch (memError) {
							// Error handling for memory operations if needed, but no logging requested
						}
					}
				}
			}

			if (memory && outputParser && response.output) {
				try {
					const parsedOutput = jsonParse<{ output: Record<string, unknown> }>(
						response.output as string,
					);
					// Stringify the parsed output to satisfy type string
					response.output = JSON.stringify(parsedOutput?.output ?? parsedOutput);
				} catch (e) {
					// Error handling for parsing if needed, but no logging requested
				}
			}

			const baseResult = omit(
				response,
				'system_message',
				'formatting_instructions',
				'input',
				'chat_history',
				'agent_scratchpad',
			) as Record<string, any>;
			if (options.outputTokensConsumption && tokenCallbackHandler.getUsage().totalTokens > 0) {
				const tokenUsage = tokenCallbackHandler.getUsage();
				baseResult.promptTokens = tokenUsage.promptTokens;
				baseResult.completionTokens = tokenUsage.completionTokens;
				baseResult.totalTokens = tokenUsage.totalTokens;
				this.logger.debug('Token usage added to output:', tokenUsage);
			}
			const itemResult = { json: baseResult };
			returnData.push(itemResult);
		} catch (error) {
			tokenCallbackHandler.reset();
			if (this.continueOnFail()) {
				returnData.push({
					json: { error: error.message },
					pairedItem: { item: itemIndex },
				});
				continue;
			}
			throw error;
		}
	}
	return [returnData];
}
