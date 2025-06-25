// lib/handlers/playwright-loop-handler.ts
import playwright, { Page } from "playwright";
const { chromium } = playwright;
import logger from "../utils/logger";
import { computerUseLoop } from "../lib/computer-use-loop";
import { Socket } from "socket.io";
import TestScriptReviewAgent from "../agents/test-script-review-agent";
import { cua_service, CUAModelInput } from "../services/openai-cua-service";
import { LoginService } from "../services/login-service";

// Read viewport dimensions from .env file with defaults if not set
const displayWidth: number = parseInt(process.env.DISPLAY_WIDTH || "1024", 10);
const displayHeight: number = parseInt(process.env.DISPLAY_HEIGHT || "768", 10);

export async function cuaLoopHandler(
  systemPrompt: string,
  url: string,
  socket: Socket,
  testCaseReviewAgent: TestScriptReviewAgent,
  username: string,
  password: string,
  loginRequired: boolean,
  userInfo?: string
) {
  logger.info("Starting test script execution", {
    url,
    loginRequired,
    socketId: socket.id  
  });
  socket.emit("message", "Starting test script execution...");

  try {
    const browser = await chromium.launch({
      headless: false,
      env: {},
      args: ["--disable-extensions", "--disable-file-system"],
    });

    logger.debug("Browser launched successfully");
    socket.emit("message", "Launching browser...");

    const page = await browser.newPage();

    // Set the page as data in the socket.
    socket.data.page = page;

    // Set viewport dimensions using env values
    await page.setViewportSize({ width: displayWidth, height: displayHeight });
    logger.debug(`Viewport set:\n${JSON.stringify({ width: displayWidth, height: displayHeight }, null, 2)}`);

    // Navigate to the provided URL from the form.
    await page.goto(url);
    logger.debug(`Navigated to URL: ${url}`);

    // wait for 2 seconds
    await page.waitForTimeout(2000);

    // Capture an initial screenshot.
    const screenshot_before_login = await page.screenshot();
    const screenshot_before_login_base64 = screenshot_before_login.toString("base64");

    logger.debug("Initial screenshot captured");

    // Asynchronously check the status of the test script.
    const testScriptReviewResponsePromise =
      testCaseReviewAgent.checkTestScriptStatus(screenshot_before_login_base64);

    // Asynchronously emit the test script review response to the socket.
    testScriptReviewResponsePromise.then((testScriptReviewResponse) => {
      logger.debug("Sending initial screenshot to TestScriptReviewAgent");
      socket.emit("testscriptupdate", testScriptReviewResponse);
    }).catch((error) => {
      logger.error("Initial test script review failed", { 
        error: error instanceof Error ? error.message : error 
      });
    });

    // Await till network is idle.
    await page.waitForTimeout(2000);

    let cua_model_input: CUAModelInput;

    if (loginRequired) {
      // Note to the developer: Different applications will need their own login handlers.
      logger.info("Processing login requirement");
      socket.emit("message", "Login required... proceeding with login.");

      const loginService = new LoginService();
      await loginService.fillin_login_credentials(username, password, page);

      logger.debug("Login credentials filled");

      // wait for 5 seconds
      await page.waitForTimeout(5000);

      const screenshot_after_login = await page.screenshot();
      const screenshot_after_login_base64 = screenshot_after_login.toString("base64");

      logger.debug("Post-login screenshot captured", { 
        size: screenshot_after_login_base64.length 
      });

      // Asynchronously check the status of the test script.
      const testScriptReviewResponsePromise_after_login =
        testCaseReviewAgent.checkTestScriptStatus(screenshot_after_login_base64);

      // Asynchronously emit the test script review response to the socket.
      testScriptReviewResponsePromise_after_login.then(
        (testScriptReviewResponse) => {
          logger.debug("Sending post-login screenshot to TestScriptReviewAgent");
          socket.emit("testscriptupdate", testScriptReviewResponse);
        }
      ).catch((error) => {
        logger.error("Post-login test script review failed", { 
          error: error instanceof Error ? error.message : error 
        });
      });

      await loginService.click_login_button(page);

      socket.emit("message", "Login step executed... proceeding with test script execution.");
      logger.info("Login process completed");

      cua_model_input = {
        screenshotBase64: screenshot_after_login_base64,
        previousResponseId: undefined,
        lastCallId: undefined,
      };
    } else {
      // If login is not required, use the screenshot before login.
      logger.debug("No login required - using initial screenshot");
      cua_model_input = {
        screenshotBase64: screenshot_before_login_base64,
        previousResponseId: undefined,
        lastCallId: undefined,
      };
    }

    // Start with an initial call (without a screenshot or call_id)
    logger.debug("Setting up CUA model");
    
    const userInfoStr = userInfo ?? "";
    let initial_response = await cua_service.setupCUAModel(systemPrompt, userInfoStr);

    logger.info("CUA model setup completed", { 
      responseId: initial_response.id 
    });

    const response = await computerUseLoop(
      page,
      initial_response,
      testCaseReviewAgent,
      socket
    );

    const messageResponse = response.output.filter(
      (item: any) => item.type === "message"
    );

    if (messageResponse.length > 0) {
      messageResponse.forEach((message: any) => {
        if (Array.isArray(message.content)) {
          message.content.forEach((contentBlock: any) => {
            if (contentBlock.type === "output_text" && contentBlock.text) {
              socket.emit("message", contentBlock.text);
            }
          });
        }
      });
    }
  } catch (error) {
    logger.error("Test script execution failed", { 
      error: error instanceof Error ? error.message : error,
      url,
      loginRequired 
    });
    socket.emit("message", "Test script execution failed. Please check the logs.");
  }
}
