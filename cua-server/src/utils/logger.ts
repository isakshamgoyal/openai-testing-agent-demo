import pino from "pino";
import path from "path";
import fs from "fs";

const isProduction = process.env.NODE_ENV === "production";
const logLevel = process.env.LOG_LEVEL || "debug";

// Ensure logs directory exists
const logsDir = path.join(process.cwd(), "logs");
if (!fs.existsSync(logsDir)) {
  fs.mkdirSync(logsDir, { recursive: true });
}

// Clear existing log files on startup
const appLogPath = path.join(logsDir, "app.log");
const errorLogPath = path.join(logsDir, "error.log");

try {
  if (fs.existsSync(appLogPath)) {
    fs.writeFileSync(appLogPath, "");
  }
  if (fs.existsSync(errorLogPath)) {
    fs.writeFileSync(errorLogPath, "");
  }
} catch (error) {
  console.warn("Warning: Could not clear log files on startup:", error);
}

const logger = pino(
  {
    level: logLevel,
  },
  pino.multistream([
    // Console output (pretty in development)
    {
      level: logLevel,
      stream: isProduction
        ? process.stdout
        : pino.transport({
            target: "pino-pretty",
            options: {
              colorize: true,
            },
          }),
    },
    // File output
    {
      level: logLevel,
      stream: pino.destination({
        dest: appLogPath,
        sync: false,
      }),
    },
    // Separate error log file
    {
      level: "error",
      stream: pino.destination({
        dest: errorLogPath,
        sync: false,
      }),
    },
  ])
);

logger.info(`Logger initialized with log level: ${logLevel}`);

export default logger;
