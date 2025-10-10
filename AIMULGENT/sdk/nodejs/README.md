# AIMULGENT Node.js SDK

> Official Node.js/TypeScript client library for AIMULGENT multi-agent system

## Overview

The AIMULGENT Node.js SDK provides a convenient interface for interacting with the AIMULGENT system from Node.js and TypeScript applications. It supports code analysis, agent coordination, and system monitoring.

## Installation

```bash
npm install @aimulgent/sdk
# or
yarn add @aimulgent/sdk
# or
pnpm add @aimulgent/sdk
```

Or install from source:

```bash
git clone https://github.com/D0CT4/AMULGENT.git
cd AMULGENT/AIMULGENT/sdk/nodejs
npm install
npm run build
```

## Quick Start

### JavaScript

```javascript
const { AIMULGENTClient } = require('@aimulgent/sdk');

// Initialize client
const client = new AIMULGENTClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key' // Optional
});

// Analyze code
const code = `
function exampleFunction(x, y) {
  return x + y;
}
`;

(async () => {
  const result = await client.analyzeCode({
    code,
    language: 'javascript'
  });

  console.log(`Quality Score: ${result.qualityScore}`);
  console.log(`Issues Found: ${result.issues.length}`);

  // Get system status
  const status = await client.getSystemStatus();
  console.log(`System Status: ${status.status}`);
  console.log(`Active Agents: ${status.agentsActive}`);
})();
```

### TypeScript

```typescript
import { AIMULGENTClient, AnalyzeCodeOptions, AnalysisResult } from '@aimulgent/sdk';

// Initialize client with TypeScript types
const client = new AIMULGENTClient({
  baseUrl: 'http://localhost:8000',
  apiKey: process.env.AIMULGENT_API_KEY
});

const code = `
function exampleFunction(x: number, y: number): number {
  return x + y;
}
`;

async function analyzeCode() {
  const options: AnalyzeCodeOptions = {
    code,
    language: 'typescript',
    options: {
      checkSecurity: true,
      checkComplexity: true,
      checkStyle: true
    }
  };

  const result: AnalysisResult = await client.analyzeCode(options);
  
  result.issues.forEach(issue => {
    console.log(`[${issue.severity}] ${issue.message}`);
    console.log(`  Line ${issue.line}: ${issue.code}`);
  });
}

analyzeCode().catch(console.error);
```

## Features

### Code Analysis

```typescript
import { AIMULGENTClient } from '@aimulgent/sdk';

const client = new AIMULGENTClient({ baseUrl: 'http://localhost:8000' });

// Analyze code with detailed options
const result = await client.analyzeCode({
  code: sourceCode,
  language: 'javascript',
  options: {
    checkSecurity: true,
    checkComplexity: true,
    checkStyle: true,
    maxComplexity: 10
  }
});

// Access results
for (const issue of result.issues) {
  console.log(`[${issue.severity}] ${issue.message}`);
  console.log(`  Location: Line ${issue.line}, Column ${issue.column}`);
}
```

### System Monitoring

```typescript
// Get health status
const health = await client.getHealth();
console.log(`Status: ${health.status}`);
console.log(`Uptime: ${health.uptimeSeconds}s`);

// Get metrics
const metrics = await client.getMetrics();
console.log(`Tasks Processed: ${metrics.tasksTotal}`);
console.log(`CPU Usage: ${metrics.systemCpuPercent}%`);
```

### Agent Management

```typescript
// List available agents
const agents = await client.listAgents();
for (const agent of agents) {
  console.log(`Agent: ${agent.name} - Status: ${agent.status}`);
}

// Send task to specific agent
const result = await client.sendAgentTask({
  agentId: 'analysis-agent-001',
  taskType: 'code_review',
  payload: { code: sourceCode }
});
```

### Event Streaming

```typescript
// Subscribe to real-time events
const stream = client.subscribeToEvents();

stream.on('task.completed', (event) => {
  console.log(`Task ${event.taskId} completed`);
});

stream.on('agent.status', (event) => {
  console.log(`Agent ${event.agentId}: ${event.status}`);
});

stream.on('error', (error) => {
  console.error('Stream error:', error);
});
```

## Configuration

### Environment Variables

```bash
export AIMULGENT_BASE_URL=http://localhost:8000
export AIMULGENT_API_KEY=your-api-key
export AIMULGENT_TIMEOUT=30000
```

### Configuration File

Create `.aimulgentrc.json` in your project root:

```json
{
  "baseUrl": "http://localhost:8000",
  "apiKey": "your-api-key",
  "timeout": 30000,
  "retryAttempts": 3,
  "retryDelay": 1000
}
```

Or use `.aimulgentrc.js` for dynamic configuration:

```javascript
module.exports = {
  baseUrl: process.env.AIMULGENT_BASE_URL || 'http://localhost:8000',
  apiKey: process.env.AIMULGENT_API_KEY,
  timeout: 30000
};
```

## API Reference

### AIMULGENTClient

Main client class for interacting with AIMULGENT.

#### Constructor Options

```typescript
interface ClientOptions {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
}
```

#### Methods

- `analyzeCode(options: AnalyzeCodeOptions): Promise<AnalysisResult>`
  - Analyze code and return quality metrics

- `getSystemStatus(): Promise<SystemStatus>`
  - Get comprehensive system status

- `getHealth(): Promise<HealthStatus>`
  - Get health check information

- `getMetrics(): Promise<Metrics>`
  - Get system metrics

- `listAgents(): Promise<Agent[]>`
  - List all available agents

- `sendAgentTask(request: AgentTaskRequest): Promise<AgentTaskResult>`
  - Send task to specific agent

- `subscribeToEvents(): EventEmitter`
  - Subscribe to real-time system events

## TypeScript Support

The SDK is written in TypeScript and provides full type definitions:

```typescript
import {
  AIMULGENTClient,
  AnalyzeCodeOptions,
  AnalysisResult,
  Issue,
  Severity,
  SystemStatus,
  HealthStatus,
  Agent,
  AgentTaskRequest
} from '@aimulgent/sdk';
```

## Error Handling

```typescript
import {
  AIMULGENTClient,
  AIMULGENTError,
  ConnectionError,
  AuthenticationError,
  ValidationError,
  TimeoutError
} from '@aimulgent/sdk';

try {
  const result = await client.analyzeCode({ code, language: 'javascript' });
} catch (error) {
  if (error instanceof AuthenticationError) {
    console.error('Authentication failed:', error.message);
  } else if (error instanceof ValidationError) {
    console.error('Invalid input:', error.message);
  } else if (error instanceof ConnectionError) {
    console.error('Connection failed:', error.message);
  } else if (error instanceof TimeoutError) {
    console.error('Request timeout:', error.message);
  } else if (error instanceof AIMULGENTError) {
    console.error('AIMULGENT error:', error.message);
  }
}
```

## Examples

See the [examples](./examples/) directory for complete usage examples:

- `basic-usage.js` - Basic code analysis
- `advanced-analysis.ts` - Advanced analysis with TypeScript
- `monitoring.ts` - System monitoring and metrics
- `agent-coordination.ts` - Multi-agent workflows
- `event-streaming.ts` - Real-time event processing

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/D0CT4/AMULGENT.git
cd AMULGENT/AIMULGENT/sdk/nodejs

# Install dependencies
npm install

# Build TypeScript
npm run build

# Run tests
npm test

# Run tests with coverage
npm run test:coverage

# Lint code
npm run lint

# Format code
npm run format
```

### Project Structure

```
nodejs/
├── src/
│   ├── client.ts          # Main client class
│   ├── types.ts           # TypeScript type definitions
│   ├── errors.ts          # Error classes
│   └── utils.ts           # Utility functions
├── examples/              # Usage examples
├── tests/                 # Test files
├── package.json
├── tsconfig.json
└── README.md
```

## Contributing

Contributions are welcome! Please read our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

MIT License - see [LICENSE](../../LICENSE) for details.

## Support

- **Documentation**: [Full Documentation](https://github.com/D0CT4/AMULGENT/wiki)
- **Issues**: [GitHub Issues](https://github.com/D0CT4/AMULGENT/issues)
- **Discussions**: [GitHub Discussions](https://github.com/D0CT4/AMULGENT/discussions)
- **npm Package**: [@aimulgent/sdk](https://www.npmjs.com/package/@aimulgent/sdk)
