// Utilities for test script review functionality
import logger from "./logger";

export interface TestScriptStep {
  step_number: number;
  status: "pending" | "Pass" | "Fail";
  step_reasoning: string;
  image_path?: string;
}

export interface TestScriptState {
  steps: TestScriptStep[];
}

// Test script JSON schema for structured output
export const TEST_SCRIPT_AGENT_JSON_SCHEMA = {
    type: "object",
    properties: {
      steps: {
        type: "array",
        items: {
          type: "object",
          properties: {
            step_number: { type: "number" },
            status: {
              type: "string",
              enum: ["pending", "Pass", "Fail"]
            },
            step_reasoning: { type: "string" }
          },
          required: ["step_number", "status", "step_reasoning"],
          additionalProperties: false
        }
      }
    },
    required: ["steps"],
    additionalProperties: false
  } as const;

/**
 * Determines which steps had status changes from pending to Pass/Fail.
 */
export function getStepsWithStatusChange(oldSteps: TestScriptStep[], newSteps: TestScriptStep[]): Set<number> {
  const changedSteps = new Set<number>();
  
  oldSteps.forEach(oldStep => {
    const newStep = newSteps.find(s => s.step_number === oldStep.step_number);
    if (newStep?.status !== "pending" && oldStep.status === "pending") {
      changedSteps.add(oldStep.step_number);
    }
  });
  
  logger.debug("Status changes detected", { changedSteps: Array.from(changedSteps) });
  return changedSteps;
}

/**
 * Updates step image paths based on status changes and existing paths.
 */
export function updateStepImagePaths(
  oldSteps: TestScriptStep[], 
  newSteps: TestScriptStep[], 
  changedSteps: Set<number>, 
  screenshotPath?: string
): void {
  newSteps.forEach(newStep => {
    const oldStep = oldSteps.find(s => s.step_number === newStep.step_number);
    
    if (changedSteps.has(newStep.step_number) && screenshotPath) {
      newStep.image_path = screenshotPath;
    } else if (oldStep?.image_path) {
      newStep.image_path = oldStep.image_path;
    }
  });
} 