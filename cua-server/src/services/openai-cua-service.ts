import { AzureOpenAI } from "openai";
import logger from "../utils/logger";

// Single OpenAI client instance for CUA
const cua_client = new AzureOpenAI({
  apiKey: process.env.AZURE_API_KEY,
  apiVersion: process.env.AZURE_API_VERSION,
  endpoint: process.env.AZURE_ENDPOINT,
});

// Environment specific instructions for the CUA model
const envInstructions = process.env.ENV_SPECIFIC_INSTRUCTIONS || "";

// Display dimensions for computer use
const displayWidth: number = parseInt(process.env.DISPLAY_WIDTH || "1024", 10);
const displayHeight: number = parseInt(process.env.DISPLAY_HEIGHT || "768", 10);

// Computer use tools configuration
const CUA_TOOLS = [
  {
    type: "computer_use_preview",
    display_width: displayWidth,
    display_height: displayHeight,
    environment: "browser",
  },
  {
    type: "function",
    name: "mark_done",
    description: "Use this tool to let the user know you have finished the tasks.",
    parameters: {},
  },
];

// CUA system prompt template
const CUA_SYSTEM_PROMPT = `You are a testing agent. You will be given a list of instructions with steps to test a web application. 
You will need to navigate the web application and perform the actions described in the instructions.
Try to accomplish the provided task in the simplest way possible.
Once you believe your are done with all the tasks required or you are blocked and cannot progress
(for example, you have tried multiple times to acommplish a task but keep getting errors or blocked),
use the mark_done tool to let the user know you have finished the tasks.
You do not need to authenticate on user's behalf, the user will authenticate and your flow starts after that.`;

// CUA-specific interfaces
export interface CUAModelInput {
  screenshotBase64: string;
  previousResponseId?: string;
  lastCallId?: string;
}

export interface CUAModelResponse {
  id: string;
  output: Array<any>;
}

class OpenAICUAService {
  private readonly model: string;

  constructor() {
    this.model = process.env.AZURE_COMPUTER_USE_MODEL_DEPLOYMENT_NAME || 'computer-use-preview';
  }

  /**
   * Initialize CUA model with system and user prompts
   */
  async setupCUAModel(systemPrompt: string, userInfo: string): Promise<CUAModelResponse> {
    logger.debug("Setting up CUA model", { 
      systemPromptLength: systemPrompt.length,
      userInfoLength: userInfo.length 
    });
    
    const enhancedSystemPrompt = `${CUA_SYSTEM_PROMPT}
      ${envInstructions ? `Environment specific instructions: ${envInstructions}` : ""}`;

    const input = [
      {
        role: "system",
        content: enhancedSystemPrompt,
      },
      {
        role: "user",
        content: `INSTRUCTIONS:\n${systemPrompt}\n\nUSER INFO:\n${userInfo}`,
      },
    ];

    return this.callCUAModel(input);
  }

  /**
   * Send screenshot and user input to CUA model
   */
  async sendScreenshotToModel(
    { screenshotBase64, previousResponseId, lastCallId }: CUAModelInput,
    userMessage?: string
  ): Promise<CUAModelResponse> {
    logger.debug("Sending screenshot to CUA model", {
      screenshotSize: screenshotBase64.length,
      hasCallId: !!lastCallId,
      hasUserMessage: !!userMessage,
      hasPreviousResponse: !!previousResponseId
    });

    const input: any[] = [];

    if (lastCallId) {
      input.push({
        call_id: lastCallId,
        type: "computer_call_output",
        output: {
          type: "input_image",
          image_url: `data:image/png;base64,${screenshotBase64}`,
        },
      });
    }

    if (userMessage) {
      input.push({
        role: "user",
        content: userMessage,
      });
    }

    return this.callCUAModel(input, previousResponseId);
  }

  /**
   * Send function call result back to CUA model
   */
  async sendFunctionResult(
    callId: string,
    previousResponseId: string,
    resultData: object = {}
  ): Promise<CUAModelResponse> {
    logger.debug("Sending function result to CUA model", {
      callId,
      previousResponseId,
      resultData
    });

    const input = [
      {
        call_id: callId,
        type: "function_call_output",
        output: JSON.stringify(resultData),
      },
    ];

    return this.callCUAModel(input, previousResponseId);
  }

  /**
   * Core CUA model call with proper request configuration
   */
  async callCUAModel(input: any[], previousResponseId?: string): Promise<CUAModelResponse> {
    logger.debug("Sending request to CUA model");

    const requestBody: any = {
      model: this.model,
      tools: CUA_TOOLS,
      input,
      reasoning: {
        generate_summary: "concise",
      },
      truncation: "auto",
      tool_choice: "required",
    };

    if (previousResponseId) {
      requestBody.previous_response_id = previousResponseId;
      logger.debug("Including previous response ID", { previousResponseId });
    }

    try {
      const response = await cua_client.responses.create(requestBody);
      logger.debug("CUA model response received", { responseId: response.id });
      return response as CUAModelResponse;
    } catch (error) {
      logger.error("CUA model call failed", { error: error instanceof Error ? error.message : error });
      throw new Error(`CUA model call failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
  
}

// Export singleton instance
export const cua_service = new OpenAICUAService();