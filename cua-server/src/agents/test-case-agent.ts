import { PROMPT_WITHOUT_LOGIN, PROMPT_WITH_LOGIN } from "../lib/constants";
import logger from "../utils/logger";
import { openai_service } from "../services/openai-service";
import { TestCase, TEST_CASE_JSON_SCHEMA } from "../utils/test-case-utils";

class TestCaseAgent {
  private readonly model = process.env.AZURE_TEST_CASE_AGENT_DEPLOYMENT_NAME_CHAT || "o3-mini";
  private readonly system_prompt: string;
  private readonly login_required: boolean;

  constructor(login_required = false) {
    this.login_required = login_required;
    this.system_prompt = login_required ? PROMPT_WITH_LOGIN : PROMPT_WITHOUT_LOGIN;
  }

  /**
   * Generate test steps via the unified Response API.
   */
  async generateTestCases(userInstruction: string): Promise<TestCase> {
    logger.info("Generating Test Cases");
    
    const response = await openai_service.responseAPI({
      systemPrompt: this.system_prompt,
      userMessage: userInstruction,
      model: this.model,
      schema: TEST_CASE_JSON_SCHEMA,
      schemaName: "test_case"
    });

    if (!response.output_text) {
      throw new Error("No output text received from OpenAI service");
    }

    const result: TestCase = JSON.parse(response.output_text);

    logger.info(`Test Cases Generated Successfully:\n${JSON.stringify({ 
      loginRequired: this.login_required,
      steps: result.steps
    }, null, 2)}`);
    
    return result;
  }
}

export default TestCaseAgent;
