# MCP Gateway Plugin System

The MCP Gateway includes a flexible plugin system that allows for extending functionality through custom plugins. This document explains how to create and use plugins with the gateway.

## Plugin Types

The gateway supports two main types of plugins:

1. **Guardrail Plugins**: These plugins can modify or block requests and responses based on security or compliance rules.
2. **Tracing Plugins**: These plugins observe requests and responses for logging, monitoring, or auditing purposes.

## Creating Custom Plugins

To create a custom plugin, follow these steps:

1. Create a new Python file in the appropriate subdirectory (`guardrails/` or `tracing/`).
2. Extend the appropriate base class (`GuardrailPlugin` or `TracingPlugin`).
3. Implement the required methods.
4. Set the `plugin_name` attribute for easy identification and loading.

### Example Plugin

```python
from typing import Any, Dict, Optional
from mcp_gateway.plugins.base import GuardrailPlugin, PluginContext

class MyCustomGuardrailPlugin(GuardrailPlugin):
    """A custom guardrail plugin that does something useful."""
    
    plugin_type = "guardrail"  # Must match the plugin type
    plugin_name = "my-custom"  # Used for identification in configuration
    
    def __init__(self):
        # Initialize your plugin
        pass
        
    def load(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Load plugin configuration."""
        # Handle configuration here
        pass
        
    def process_request(self, context: PluginContext) -> Optional[Dict[str, Any]]:
        """Process the request."""
        # Modify or validate request arguments
        return context.arguments
        
    def process_response(self, context: PluginContext) -> Any:
        """Process the response."""
        # Modify or validate the response
        return context.response
```

## Plugin Configuration

Plugins are configured and loaded through command-line arguments or configuration files. The `plugin_name` attribute is used to identify which plugins to load.

### Command-line Example

```bash
python -m mcp_gateway.server --enable-guardrails my-custom,basic --enable-tracing basic
```

### JSON Configuration Example

```json
{
  "mcpServers": {
    "mcp-gateway": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-proxy",
        "run",
        "mcp_gateway/server.py",
        "--enable-guardrails", "my-custom,basic",
        "--enable-tracing", "basic"
      ],
      "env": {
        "PYTHONPATH": "/path/to/mcp-proxy"
      }
    }
  }
}
```

## Plugin Discovery

The plugin manager scans subdirectories under the `plugins/` directory to discover available plugins. Plugins are identified by their `plugin_name` attribute, which makes it easier to reference them in configuration.

The `plugin_name` should be:
- Unique among plugins of the same type
- Lowercase and simple (can contain hyphens for readability)
- Reflective of the plugin's purpose

## Built-in Plugins

The gateway comes with several built-in plugins:

- **`basic`** (Guardrail): A basic guardrail that anonymizes PII and removes common secrets
- **`lasso`** (Guardrail): Integrates with Lasso Security's API for content security
- **`basic`** (Tracing): A simple tracing plugin that logs requests and responses
