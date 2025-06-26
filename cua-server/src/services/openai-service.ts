import { OpenAI, AzureOpenAI } from "openai";
import logger from "../utils/logger";

// If USE_OPENAI=true, use OpenAI, otherwise use Azure-OpenAI
const client = process.env.USE_OPENAI === 'true' 
  ? new OpenAI({
      apiKey: process.env.OPENAI_API_KEY,
    })
  : new AzureOpenAI({
      apiKey: process.env.AZURE_API_KEY,
      apiVersion: process.env.AZURE_API_VERSION,
      endpoint: process.env.AZURE_ENDPOINT,
    });

// Unified OpenAI service interface
export interface ResponseAPIInput {
    systemPrompt: string;
    userMessage?: string;
    base64Image?: string;
    previousResponseId?: string;
    model?: string;
    schema?: object;
    schemaName?: string;
}

export interface ResponseAPIResponse {
  id: string;
  output: Array<any>;
  output_text?: string;
  output_parsed?: any;
}

class OpenAIService {
  private buildUserContent(userMessage?: string, base64Image?: string): any {
    // If we have both text and image, create multimodal content
    if (userMessage && base64Image) {
      return [
        {
          type: "input_text",
          text: userMessage,
        },
        {
          type: "input_image",
          image_url: `data:image/png;base64,${base64Image}`,
          detail: "high",
        }
      ];
    }

    // If only text
    if (userMessage) {
      return userMessage;
    }

    // If only image
    if (base64Image) {
      return [
        {
          type: "input_image",
          image_url: `data:image/png;base64,${base64Image}`,
          detail: "high",
        }
      ];
    }

    return userMessage;
  }

  /**
   * Unified response API call that handles all scenarios
   */
  async responseAPI({ 
    systemPrompt, 
    userMessage, 
    base64Image,
    previousResponseId, 
    model,
    schema,
    schemaName
  }: ResponseAPIInput): Promise<ResponseAPIResponse> {

    const operationName = schema ? `${schemaName} Agent` : "Basic Response";  
    logger.debug(`OpenAI ResponseAPI called for: ${operationName}`);
    logger.trace(`OpenAI ResponseAPI called with input:\n${JSON.stringify({
      systemPrompt: systemPrompt,
      userMessage: userMessage,
      hasBase64Image: !!base64Image,
      previousResponseId: previousResponseId || null,
      model: model || null,
      schemaName: schemaName || null,
    }, null, 2)}`);

    const userContent = this.buildUserContent(userMessage, base64Image);

    const requestBody: any = {
      model: model,
      input: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userContent },
      ],
    };

    // Add JSON schema if provided
    if (schema) {
      requestBody.text = {
        format: {
          type: "json_schema",
          name: schemaName || "response_output",
          schema: schema
        }
      };
    }

    // Add previous response ID if provided
    if (previousResponseId) {
      requestBody.previous_response_id = previousResponseId;
    }

    logger.trace(`Request Body:\n${JSON.stringify(requestBody, null, 2)}`);

    try {
      // First attempt
      const response = await client.responses.create(requestBody);
      logger.debug(`${operationName} Completed Successfully!\n${JSON.stringify({ 
        responseId: response.id,
        model: model
      }, null, 2)}`);
      return response as ResponseAPIResponse;
    } catch (error) {
      logger.error(`${operationName} failed\n${JSON.stringify({ 
        error: error instanceof Error ? error.message : error,
        model: model,
        errorType: error instanceof Error ? error.constructor.name : 'Unknown'
      }, null, 2)}`);

      // Retry without previousResponseId if it was provided
      if (previousResponseId) {
        logger.debug("Retrying without PreviousResponseId...");
        delete requestBody.previous_response_id;
        
        try {
          const retryResponse = await client.responses.create(requestBody);
          logger.debug(`${operationName} retry succeeded\n${JSON.stringify({ 
            responseId: retryResponse.id,
            model: model
          }, null, 2)}`);
          return retryResponse as ResponseAPIResponse;
        } catch (retryError) {
          logger.error(`${operationName} retry also failed\n${JSON.stringify({ 
            error: retryError instanceof Error ? retryError.message : retryError,
            model: model
          }, null, 2)}`);
        }
      }
      
      throw error;
    }
  }
}

export const openai_service = new OpenAIService(); 