import fs from "fs";
import path from "path";
import { v4 as uuidv4 } from "uuid";
import logger from "./logger";

export class ScreenshotUtils {
  private runFolder: string;

  constructor(runFolder?: string) {
    this.runFolder = runFolder || uuidv4();
  }

  /**
   * Gets the current run folder ID.
   */
  getRunFolder(): string {
    return this.runFolder;
  }

  /**
   * Sets the run folder ID.
   */
  setRunFolder(runFolder: string): void {
    this.runFolder = runFolder;
  }

  /**
   * Ensures the run folder exists, creating it if necessary.
   */
  async ensureRunFolder(): Promise<void> {
    const runFolderPath = this.getRunFolderPath();
    
    if (!fs.existsSync(runFolderPath)) {
      fs.mkdirSync(runFolderPath, { recursive: true });
      logger.debug("Run folder created", { path: runFolderPath });
    }
  }

  /**
   * Gets the full path to the run folder.
   */
  getRunFolderPath(): string {
    return path.join(process.cwd(), "..", "frontend", "public", "test_results", this.runFolder);
  }

  /**
   * Saves a screenshot and returns the relative path.
   */
  async saveScreenshot(base64Image: string): Promise<string> {
    await this.ensureRunFolder();
    
    const screenshotFilename = `${uuidv4()}.png`;
    const screenshotPathLocal = path.join(this.getRunFolderPath(), screenshotFilename);
    const relativePath = `/test_results/${this.runFolder}/${screenshotFilename}`;
    
    fs.writeFileSync(screenshotPathLocal, Buffer.from(base64Image, "base64"));
    logger.debug("Screenshot saved", { relativePath });
    
    return relativePath;
  }
} 