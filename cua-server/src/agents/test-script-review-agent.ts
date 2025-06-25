/**
 * This agent processes test script review tasks sequentially using a task queue.
 * Each call to checkTestScriptStatus enqueues a new screenshot processing job.
 */
import logger from "../utils/logger";
import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";
import { TEST_SCRIPT_REVIEW_PROMPT } from "../lib/constants";
import { openai_service } from "../services/openai-service";
import { ScreenshotUtils } from "../utils/screenshot-utils";
import { TestScriptState, getStepsWithStatusChange, updateStepImagePaths, TEST_SCRIPT_AGENT_JSON_SCHEMA } from "../utils/test-script-utils";

interface ReviewTask {
  base64Image: string;
  userInstruction?: string;
  resolve: (value: string) => void;
  reject: (error: Error) => void;
}

class TestScriptReviewAgent {
  private readonly model: string;
  private previous_response_id: string | null = null;
  private test_script_state: TestScriptState | null = null;
  private screenshotUtils: ScreenshotUtils | null = null;

  // Flag whether to include the previous screenshot response in the input to the LLM - true works best
  private readonly includePreviousResponse: boolean = true;

  // Task queue related properties
  private readonly taskQueue: ReviewTask[] = [];
  private isProcessingQueue: boolean = false;

  constructor() {
    // Set the default model to use Azure deployment
    this.model = process.env.AZURE_TEST_SCRIPT_REVIEW_AGENT_DEPLOYMENT_NAME_CHAT || 'gpt-4o';
    logger.debug(`TestScriptReviewAgent Initialized with model: ${this.model}`);
  }

  /**
   * Creates the initial test script state from the user instructions.
   */
  async instantiateAgent(userInstruction: string): Promise<string> {
    logger.trace(`Instantiation agent with instruction (This should only be called once per test script run):\n${userInstruction}`);

    try {
      const response = await openai_service.responseAPI({
        systemPrompt: TEST_SCRIPT_REVIEW_PROMPT,
        userMessage: "Instructions: " + userInstruction,
        model: this.model,
        schema: TEST_SCRIPT_AGENT_JSON_SCHEMA,
        schemaName: "test_script_output"
      });

      logger.info(`Agent Instantiation Successful: ${response.id}`);

      this.previous_response_id = response.id;
      this.test_script_state = JSON.parse(response.output_text!) as TestScriptState;
      
      // Create screenshot utils instance for this session
      this.screenshotUtils = new ScreenshotUtils();
      await this.screenshotUtils.ensureRunFolder();

      return response.output_text!;
    } catch (error) {
      logger.error(`Failed to Instantiate Agent:\n${JSON.stringify({ 
        error: error instanceof Error ? error.message : error 
      }, null, 2)}`);
      throw new Error(`Failed to instantiate agent: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Enqueues a new test script review task.
   */
  async checkTestScriptStatus(base64Image: string, userInstruction?: string): Promise<string> {
    logger.debug("Enqueuing test script review task");

    return new Promise((resolve, reject) => {
      this.taskQueue.push({ base64Image, userInstruction, resolve, reject });
      this.processQueue();
    });
  }

  /**
   * Processes the task queue sequentially.
   */
  private async processQueue(): Promise<void> {
    if (this.isProcessingQueue) {
      logger.debug("Queue processing already in progress");
      return;
    }
    
    this.isProcessingQueue = true;
    logger.debug("Starting Queue Processing");

    try {
      while (this.taskQueue.length > 0) {
        const task = this.taskQueue.shift()!;
        try {
          const result = await this.processTestScriptReview(task.base64Image, task.userInstruction);
          task.resolve(result);
        } catch (error) {
          logger.error(`Task processing failed:\n${JSON.stringify({ error: error instanceof Error ? error.message : error }, null, 2)}`);
          task.reject(error instanceof Error ? error : new Error(String(error)));
        }
      }
    } finally {
      this.isProcessingQueue = false;
      logger.debug("Queue processing completed");
    }
  }

  /**
   * Processes the test script status by sending the screenshot (and optional instruction) to the LLM,
   * then updating the test script state with any changes.
   */
  private async processTestScriptReview(base64Image: string, userInstruction?: string): Promise<string> {
    logger.debug(`Processing TestScriptReviewAgent with Previous Response ID: ${this.previous_response_id}`);
  
    // If we don't already have a test_script_state, just parse blank structure
    if (!this.test_script_state) {
      this.test_script_state = { steps: [] };
      logger.warn("No previous test script state found, creating empty state");
    }

    if (!this.screenshotUtils) {
      throw new Error("Screenshot utils not initialized. Call instantiateAgent first.");
    }

    try {
      // Call the unified AI service - just pass the schema directly!
      const response = await openai_service.responseAPI({
        systemPrompt: TEST_SCRIPT_REVIEW_PROMPT,
        userMessage: userInstruction ? "Context: " + userInstruction : undefined,
        base64Image,
        previousResponseId: this.previous_response_id || undefined,
        model: this.model,
        schema: TEST_SCRIPT_AGENT_JSON_SCHEMA,
        schemaName: "test_script_output"
      });

      logger.debug(`TestScriptReviewAgent Response received with Response ID: ${response.id}`);

      // Update previous response id if configured to do so
      if (this.includePreviousResponse) {
        this.previous_response_id = response.id;
      }

      // Parse the new state
      const newState: TestScriptState = JSON.parse(response.output_text!);
      const oldSteps = this.test_script_state.steps;
      
      // Determine if any steps changed status
      const changedSteps = getStepsWithStatusChange(oldSteps, newState.steps);
      
      // Save screenshot if there were status changes
      let screenshotPath: string | undefined;
      if (changedSteps.size > 0) {
        screenshotPath = await this.screenshotUtils.saveScreenshot(base64Image);
      }

      // Update image paths for all steps
      updateStepImagePaths(oldSteps, newState.steps, changedSteps, screenshotPath);

      // Update internal state
      this.test_script_state = newState;

      const updatedJson = JSON.stringify(this.test_script_state);
      logger.debug(`Test Script State Updated:\n${JSON.stringify({ 
        stepsCount: this.test_script_state.steps.length,
        changedStepsCount: changedSteps.size
      }, null, 2)}`);
      return updatedJson;

    } catch (error) {
      logger.error("Test script review processing failed", { 
        error: error instanceof Error ? error.message : error 
      });
      throw new Error(`Failed to process test script status: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

export default TestScriptReviewAgent;
