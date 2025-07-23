import logger from "../utils/logger";
import { Socket } from "socket.io";
import { cua_service, CUAModelInput } from "../services/openai-cua-service";
import { computerUseLoop } from "../lib/computer-use-loop";

export async function handleSocketMessage(
  socket: Socket,
  msg: string
): Promise<void> {
  logger.debug("Handling socket message", { 
    messageLength: msg.length,
    socketId: socket.id 
  });

  // A message from user resumes the test script or instructs model to take an action.
  const page = socket.data.page;
  const previousResponseId = socket.data.previousResponseId;
  const testCaseReviewAgent = socket.data.testCaseReviewAgent;

  const screenshot = await page.screenshot();
  const screenshotBase64 = screenshot.toString("base64");

  const lastCallId = socket.data.lastCallId;
  const modelInput: CUAModelInput = {
    screenshotBase64: screenshotBase64,
    previousResponseId: previousResponseId,
    lastCallId: lastCallId,
  };

  logger.debug("Sending input to CUA model");

  const resumeResponse = await cua_service.sendScreenshotToModel(modelInput, msg);

  const response = await computerUseLoop(
    page,
    resumeResponse,
    testCaseReviewAgent,
    socket
  );

  const messageResponse = response.output.filter(
    (item: any) => item.type === "message"
  );

  if (messageResponse.length > 0) {
    logger.debug("Emitting model messages to socket");
    
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
}
