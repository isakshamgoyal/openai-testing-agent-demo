# CUA Server - Azure OpenAI Edition

This is the core testing agent server that communicates with Azure OpenAI's computer-use-preview model to drive Playwright automation.

## Prerequisites

- Azure OpenAI resource with access to the computer-use-preview model
- Node.js and npm
- Valid Azure OpenAI API key and endpoint

## Azure OpenAI Setup

1. **Create Azure OpenAI Resource**: Set up an Azure OpenAI resource in your Azure portal
2. **Deploy Models**: Deploy the following models in your Azure OpenAI resource:
   - `computer-use-preview` (for computer use capabilities)
   - `gpt-4o` (for test case generation and review)
3. **Get API Key and Endpoint**: Obtain your API key and endpoint from the Azure portal

## Configuration

1. **Copy environment files**:
   ```bash
   cp .env.example .env.development
   # edit .env.development
   ```

2. **Configure Azure OpenAI settings** in `.env.development`:
   ```bash
   # Your Azure OpenAI API key
   AZURE_API_KEY=your_azure_openai_api_key_here
   
   # Your Azure OpenAI endpoint
   AZURE_ENDPOINT=https://your-resource-name.openai.azure.com
   
   # API version (use the latest preview version)
   AZURE_API_VERSION=2025-04-01-preview
   
   # Deployment names from your Azure OpenAI resource
   AZURE_DEPLOYMENT_NAME=computer-use-preview
   AZURE_DEPLOYMENT_NAME_CHAT=gpt-4o
   ```

3. **Optional configurations**:
   ```bash
   # Display dimensions for the browser automation
   DISPLAY_WIDTH=1024
   DISPLAY_HEIGHT=768
   
   # Server configuration
   SOCKET_PORT=8000
   CORS_ORIGIN=*
   
   # Logging level
   LOG_LEVEL=info
   
   # Environment-specific instructions (e.g., for macOS)
   ENV_SPECIFIC_INSTRUCTIONS=Use CMD key instead of CTRL key for macOS
   ```

## Usage

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Run in development mode**:
   ```bash
   npm run dev
   ```

3. **Build and run**:
   ```bash
   npm run build
   npm start
   ```

## Azure OpenAI API Compatibility

This server now uses Azure OpenAI API instead of the standard OpenAI API. The key differences:

- Uses `AzureOpenAI` client instead of `OpenAI`
- Requires endpoint and API version configuration
- Uses deployment names instead of model names
- Supports the same computer-use-preview functionality via Azure

## API Example

The server makes requests similar to this curl example:
```bash
curl -X POST "https://your-resource-name.openai.azure.com/openai/responses?api-version=2025-04-01-preview" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AZURE_API_KEY" \
  -d '{
     "model": "computer-use-preview",
     "input": [...]
    }'
```

## Environment Variables Reference

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AZURE_API_KEY` | Your Azure OpenAI API key | Yes | - |
| `AZURE_ENDPOINT` | Your Azure OpenAI endpoint URL | Yes | - |
| `AZURE_API_VERSION` | Azure OpenAI API version | No | `2025-04-01-preview` |
| `AZURE_DEPLOYMENT_NAME` | Computer-use model deployment name | No | `computer-use-preview` |
| `AZURE_DEPLOYMENT_NAME_CHAT` | Chat model deployment name | No | `gpt-4o` |
| `DISPLAY_WIDTH` | Browser viewport width | No | `1024` |
| `DISPLAY_HEIGHT` | Browser viewport height | No | `768` |
| `SOCKET_PORT` | WebSocket server port | No | `8000` |
| `CORS_ORIGIN` | CORS origin setting | No | `*` |
| `LOG_LEVEL` | Logging level | No | `info` |
| `ENV_SPECIFIC_INSTRUCTIONS` | OS-specific instructions | No | - |

## Architecture

The server includes several key components:

- **OpenAI CUA Client** (`services/openai-cua-client.ts`): Handles Azure OpenAI API communication
- **Test Case Agent** (`agents/test-case-agent.ts`): Generates test case steps
- **Test Script Review Agent** (`agents/test-script-review-agent.ts`): Reviews and updates test progress
- **Computer Use Loop** (`lib/computer-use-loop.ts`): Main automation loop
- **Handlers**: WebSocket and browser automation handlers

## Troubleshooting

1. **Authentication Error**: Verify your `AZURE_API_KEY` and `AZURE_ENDPOINT` are correct
2. **Model Not Found**: Ensure your deployment names match what's configured in Azure
3. **API Version Error**: Try using a different `AZURE_API_VERSION` if the preview version is not available
4. **Rate Limiting**: Azure OpenAI has rate limits that may be different from standard OpenAI

## Migration from OpenAI

This project has been updated from standard OpenAI to Azure OpenAI. Key changes:

1. `OpenAI` client replaced with `AzureOpenAI`
2. `OPENAI_API_KEY` replaced with `AZURE_API_KEY`
3. Added `AZURE_ENDPOINT` and `AZURE_API_VERSION` requirements
4. Model names replaced with deployment names
5. All API calls now go through Azure OpenAI endpoints
