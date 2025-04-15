# server.py
import asyncio
import logging
import os
import json
import argparse
import sys
from contextlib import asynccontextmanager, AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, AsyncIterator, List, Optional, Tuple

from mcp.server.fastmcp import FastMCP, Context
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

from mcp_gateway.config import load_config
from mcp_gateway.sanitizers import (
    SanitizationError,
    sanitize_tool_call_args,
    sanitize_tool_call_result,
    sanitize_resource_read,
    sanitize_response,
)
from mcp_gateway.plugins.manager import PluginManager

# --- Global Config for Args ---
cli_args = None
log_level = os.environ.get("LOGLEVEL", "INFO").upper()

# Configure logging
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Server:
    """Manages the connection and interaction with a single proxied MCP server."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initializes the Proxied Server.

        Args:
            name: The unique name identifier for this server.
            config: The configuration dictionary for this server (command, args, env).
        """
        self.name = name
        self.config = config
        self._session: Optional[ClientSession] = None
        self._client_cm: Optional[
            AsyncIterator[Tuple[asyncio.StreamReader, asyncio.StreamWriter]]
        ] = None
        self._server_info: Optional[types.InitializeResult] = None
        self._exit_stack = AsyncExitStack()
        logger.info(f"Initialized Proxied Server: {self.name}")

    @property
    def session(self) -> ClientSession:
        """Returns the active ClientSession, raising an error if not started."""
        if self._session is None:
            raise RuntimeError(f"Server '{self.name}' session not started.")
        return self._session

    async def start(self) -> None:
        """Starts the underlying MCP server process and establishes a client session."""
        if self._session is not None:
            logger.warning(f"Server '{self.name}' already started.")
            return

        logger.info(f"Starting proxied server: {self.name}...")
        try:
            server_params = StdioServerParameters(
                command=self.config.get("command", ""),
                args=self.config.get("args", []),
                env=self.config.get("env", None),
            )

            # Use AsyncExitStack to manage the stdio_client context
            self._client_cm = stdio_client(server_params)
            read, write = await self._exit_stack.enter_async_context(self._client_cm)

            # Use AsyncExitStack to manage the ClientSession context
            session_cm = ClientSession(read, write)
            self._session = await self._exit_stack.enter_async_context(session_cm)

            # Capture and store the InitializeResult
            self._server_info = await self._session.initialize()
            logger.info(
                f"Proxied server '{self.name}' started and initialized successfully."
            )
            # Optionally log server info/capabilities if needed for debugging
            # logger.debug(f"Server '{self.name}' info: {self._server_info}")

        except Exception as e:
            logger.error(f"Failed to start server '{self.name}': {e}", exc_info=True)
            self._server_info = None  # Ensure server_info is None on failure
            await self.stop()  # Attempt cleanup if start failed
            raise

    async def stop(self) -> None:
        """Stops the underlying MCP server process and closes the client session."""
        logger.info(f"Stopping proxied server: {self.name}...")
        await self._exit_stack.aclose()
        self._session = None
        self._client_cm = None
        self._server_info = None  # Clear server info on stop
        logger.info(f"Proxied server '{self.name}' stopped.")

    # --- MCP Interaction Methods ---

    async def list_prompts(self) -> List[types.Prompt]:
        """Lists available prompts from the proxied server."""
        # No sanitization needed for listing generally
        return await self.session.list_prompts()

    async def get_prompt(
        self,
        plugin_manager: PluginManager,
        name: str,
        arguments: Optional[Dict[str, str]] = None,
        mcp_context: Optional[Any] = None,
    ) -> types.GetPromptResult:
        """Gets a specific prompt from the proxied server, processing through plugins."""
        logger.info(f"Getting prompt {name} with arguments {arguments}")

        # Use original arguments for the actual call
        result = await self.session.get_prompt(name, arguments=arguments)

        # 2. Sanitize Response
        sanitized_result = await sanitize_response(
            plugin_manager=plugin_manager,
            server_name=self.name,
            capability_type="prompt",
            name=name,
            response=result,
            request_arguments=arguments,
            mcp_context=mcp_context,
        )

        # Ensure the result is still the correct type
        if isinstance(sanitized_result, types.GetPromptResult):
            return sanitized_result
        else:
            logger.error(
                f"Response plugin for prompt {name} returned unexpected type {type(sanitized_result)}. Returning original."
            )
            return result

    async def list_resources(self) -> List[types.Resource]:
        """Lists available resources from the proxied server."""
        # No sanitization needed for listing generally
        return await self.session.list_resources()

    async def read_resource(
        self,
        plugin_manager: PluginManager,
        uri: str,
        mcp_context: Optional[Any] = None,
    ) -> Tuple[bytes, Optional[str]]:
        """Reads a resource from the proxied server after processing through plugins."""
        # No request args to sanitize for read_resource itself

        content, mime_type = await self.session.read_resource(uri)

        # Sanitize the response content using the dedicated function
        sanitized_content, sanitized_mime_type = await sanitize_resource_read(
            plugin_manager=plugin_manager,
            server_name=self.name,
            uri=uri,
            content=content,
            mime_type=mime_type,
            mcp_context=mcp_context,
        )
        return sanitized_content, sanitized_mime_type

    async def list_tools(self) -> List[types.Tool]:
        """Lists available tools from the proxied server."""
        # No sanitization needed for listing generally
        return await self.session.list_tools()

    async def call_tool(
        self,
        plugin_manager: PluginManager,
        name: str,
        arguments: Optional[Dict[str, Any]] = None,
        mcp_context: Optional[Any] = None,
    ) -> types.CallToolResult:
        """Calls a tool on the proxied server after processing args and result through plugins."""
        # 1. Sanitize request arguments
        sanitized_args = await sanitize_tool_call_args(
            plugin_manager=plugin_manager,
            server_name=self.name,
            tool_name=name,
            arguments=arguments,
            mcp_context=mcp_context,
        )

        if sanitized_args is None:
            # Handle blocked request appropriately
            logger.warning(
                f"Tool call {self.name}/{name} blocked by request sanitizer plugin."
            )
            # Return an error result structured for CallToolResult
            return types.CallToolResult(
                outputs=[
                    {
                        "type": "error",
                        "message": f"Request blocked by gateway policy for tool '{name}'.",
                    }
                ]
            )

        # 2. Call the tool with sanitized arguments
        result = await self.session.call_tool(name, arguments=sanitized_args)

        # 3. Sanitize the response result
        # Pass original request arguments for context if needed by plugins
        sanitized_result = await sanitize_tool_call_result(
            plugin_manager=plugin_manager,
            server_name=self.name,
            tool_name=name,
            result=result,
            request_arguments=arguments,
            mcp_context=mcp_context,
        )

        return sanitized_result

    async def get_capabilities(self) -> Optional[types.ServerCapabilities]:
        """Gets the capabilities of the proxied server from the stored InitializeResult."""
        if self._server_info is None:
            logger.warning(
                f"Server '{self.name}' InitializeResult not available (initialization failed or pending?)."
            )
            return None
        if self._server_info.capabilities is None:
            # MCP spec says capabilities is required, but handle gracefully
            logger.warning(
                f"Server '{self.name}' did not report capabilities in InitializeResult."
            )
            return None
        # No sanitization typically needed for capabilities object itself
        # Plugins *could* be added here if needed (e.g., filtering reported capabilities)
        return self._server_info.capabilities


@dataclass
class GetewayContext:
    """Context holding the managed proxied servers and plugin manager."""

    proxied_servers: Dict[str, Server] = field(default_factory=dict)
    plugin_manager: Optional[PluginManager] = None


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[GetewayContext]:
    """Manages the lifecycle of proxied MCP servers and plugins."""
    global cli_args
    logger.info("MCP gateway lifespan starting...")

    # --- Plugin Manager Setup ---
    enabled_plugin_types = []
    enabled_plugins = {}

    # Process guardrail plugins
    if cli_args and cli_args.enable_guardrails:
        enabled_plugin_types.append("guardrail")
        enabled_plugins["guardrail"] = cli_args.enable_guardrails
        logger.info(f"Guardrail plugins ENABLED: {cli_args.enable_guardrails}")
    else:
        logger.info("Guardrail plugins DISABLED.")

    # Process tracing plugins
    if cli_args and cli_args.enable_tracing:
        enabled_plugin_types.append("tracing")
        enabled_plugins["tracing"] = cli_args.enable_tracing
        logger.info(f"Tracing plugins ENABLED: {cli_args.enable_tracing}")
    else:
        logger.info("Tracing plugins DISABLED by command line flag.")

    # Determine plugin directory
    plugin_dir = (
        cli_args.plugin_dir
        if cli_args and cli_args.plugin_dir
        else os.path.join(os.path.dirname(__file__), "plugins")
    )
    logger.info(f"Using plugin directory: {plugin_dir}")

    # Ensure plugin directory exists
    if not os.path.isdir(plugin_dir):
        logger.warning(
            f"Plugin directory '{plugin_dir}' not found. No plugins will be loaded."
        )
        plugin_dirs_to_scan = []
    else:
        # Scan immediate subdirectories (guardrails, tracing, etc.)
        plugin_dirs_to_scan = []
        try:
            for d in os.listdir(plugin_dir):
                dir_path = os.path.join(plugin_dir, d)
                if os.path.isdir(dir_path) and not d.startswith(
                    "_"
                ):  # Avoid __pycache__ etc.
                    plugin_dirs_to_scan.append(dir_path)
                    logger.debug(f"Added plugin directory to scan: {dir_path}")
        except Exception as e:
            logger.error(f"Error scanning plugin directories: {e}", exc_info=True)

    plugin_manager = PluginManager(plugin_dirs=plugin_dirs_to_scan)
    # TODO: Load plugin-specific configurations from a file or environment if needed
    plugin_configs = {}
    plugin_manager.discover_and_load(
        enabled_types=enabled_plugin_types,
        plugin_configs=plugin_configs,
        enabled_plugins=enabled_plugins,
    )
    # --- End Plugin Manager Setup ---

    # Load proxied server configs
    proxied_server_configs = load_config(cli_args.mcp_json_path)

    context = GetewayContext(plugin_manager=plugin_manager)

    if not proxied_server_configs:
        logger.warning(
            "No proxied MCP servers configured. Running in standalone mode (plugins still active)."
        )
    else:
        start_tasks = []
        for name, server_config in proxied_server_configs.items():
            logger.info(f"Creating client for proxied server: {name}")
            proxied_server = Server(name, server_config)
            context.proxied_servers[name] = proxied_server
            start_tasks.append(asyncio.create_task(proxied_server.start()))

        if start_tasks:
            results = await asyncio.gather(*start_tasks, return_exceptions=True)
            # Check results for errors during startup
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Find corresponding server name (requires maintaining order or mapping)
                    server_name = list(proxied_server_configs.keys())[i]
                    logger.error(
                        f"Failed to start server '{server_name}' during gather: {result}",
                        exc_info=result,
                    )
                    # Optionally remove failed server from context?
                    # context.proxied_servers.pop(server_name, None)
            logger.info("Attempted to start all configured proxied servers.")

    try:
        yield context
    finally:
        logger.info("MCP gateway lifespan shutting down...")
        stop_tasks = [
            asyncio.create_task(server.stop())
            for server in context.proxied_servers.values()
            if server._session is not None  # Only stop started servers
        ]
        if stop_tasks:
            await asyncio.gather(*stop_tasks)
            logger.info("All active proxied servers stopped.")
        logger.info("MCP gateway shutdown complete.")


# Initialize the MCP gateway server
# Pass description and version if desired
mcp = FastMCP("MCP Gateway", lifespan=lifespan, version="0.1.0")


@mcp.tool()
async def get_metadata(ctx: Context) -> Dict[str, Any]:
    """Provides metadata about all available proxied MCPs via a tool call."""
    geteway_context: GetewayContext = ctx.request_context.lifespan_context
    metadata: Dict[str, Any] = {}

    if not geteway_context.proxied_servers:
        return {"status": "standalone_mode", "message": "No proxied MCPs configured"}

    for name, server in geteway_context.proxied_servers.items():
        server_metadata: Dict[str, Any] = {
            "capabilities": None,
            "tools": [],
            "resources": [],
            "prompts": [],  # Added prompts
        }
        try:
            # Ensure the session is active before fetching details
            if not server.session:
                server_metadata["error"] = "Server session not active"
                metadata[name] = server_metadata
                continue

            # 1. Get Capabilities
            capabilities = await server.get_capabilities()
            server_metadata["capabilities"] = (
                capabilities.model_dump() if capabilities else None
            )

            # 2. List Tools (only if supported)
            if capabilities and capabilities.tools:
                try:
                    tools_result = await server.list_tools()
                    # Assuming list_tools returns ListToolsResult with a 'tools' attribute
                    # Adjust if the actual return type/structure is different
                    if hasattr(tools_result, "tools"):
                        server_metadata["tools"] = [
                            tool.model_dump() for tool in tools_result.tools
                        ]
                    else:
                        # Handle unexpected result structure if necessary
                        logger.warning(
                            f"Server '{name}' list_tools returned unexpected structure: {type(tools_result)}"
                        )
                        # Attempt to use the result directly if it's already a list (might be List[Tool])
                        if isinstance(tools_result, list):
                            server_metadata["tools"] = [
                                tool.model_dump() for tool in tools_result
                            ]

                except Exception as tool_err:
                    logger.error(
                        f"Error calling list_tools on server '{name}' despite capability report: {tool_err}",
                        exc_info=True,
                    )
                    server_metadata["tools_error"] = f"Failed list_tools: {tool_err}"
            else:
                logger.info(
                    f"Server '{name}' does not support tools capability, skipping list_tools."
                )

            # 3. List Resources (only if supported)
            if capabilities and capabilities.resources:
                try:
                    resources_result = await server.list_resources()
                    # Assuming list_resources returns ListResourcesResult with a 'resources' attribute
                    # Adjust if the actual return type/structure is different
                    if hasattr(resources_result, "resources"):
                        server_metadata["resources"] = [
                            res.model_dump() for res in resources_result.resources
                        ]
                    else:
                        # Handle unexpected result structure
                        logger.warning(
                            f"Server '{name}' list_resources returned unexpected structure: {type(resources_result)}"
                        )
                        if isinstance(resources_result, list):
                            server_metadata["resources"] = [
                                res.model_dump() for res in resources_result
                            ]

                except Exception as res_err:
                    logger.error(
                        f"Error calling list_resources on server '{name}' despite capability report: {res_err}",
                        exc_info=True,
                    )
                    server_metadata["resources_error"] = (
                        f"Failed list_resources: {res_err}"
                    )
            else:
                logger.info(
                    f"Server '{name}' does not support resources capability, skipping list_resources."
                )

            # 4. List Prompts (only if supported)
            if capabilities and capabilities.prompts:
                try:
                    prompts_result = await server.list_prompts()
                    # Handle both potential return structures
                    if hasattr(prompts_result, "prompts"):
                        server_metadata["prompts"] = [
                            p.model_dump() for p in prompts_result.prompts
                        ]
                    else:
                        # Handle direct list return
                        logger.warning(
                            f"Server '{name}' list_prompts returned unexpected structure: {type(prompts_result)}"
                        )
                        if isinstance(prompts_result, list):
                            server_metadata["prompts"] = [
                                p.model_dump() for p in prompts_result
                            ]
                except Exception as prompt_err:
                    logger.error(
                        f"Error calling list_prompts on server '{name}' despite capability report: {prompt_err}",
                        exc_info=True,
                    )
                    server_metadata["prompts_error"] = (
                        f"Failed list_prompts: {prompt_err}"
                    )
            else:
                logger.info(
                    f"Server '{name}' does not support prompts capability, skipping list_prompts."
                )

            metadata[name] = server_metadata

        except Exception as e:
            # Catch general errors for this server's metadata retrieval
            logger.error(
                f"General error getting metadata for server '{name}': {e}",
                exc_info=True,
            )
            metadata[name] = {
                "error": f"Failed to retrieve metadata: {e}",
                "capabilities": server_metadata.get(
                    "capabilities"
                ),  # Include caps if fetched before error
                "tools": [],
                "resources": [],
                "prompts": [],
            }

    return metadata


@mcp.tool()
async def run_tool(
    server_name: str, tool_name: str, arguments: Dict[str, Any], ctx: Context
) -> types.CallToolResult:
    """Executes a tool on a specified proxied MCP server, running gateway plugins."""
    geteway_context: GetewayContext = ctx.request_context.lifespan_context
    proxied_server = geteway_context.proxied_servers.get(server_name)
    plugin_manager = geteway_context.plugin_manager

    if not plugin_manager:
        # Should not happen if lifespan setup is correct
        logger.error("PluginManager not found in context. Cannot execute tool.")
        return types.CallToolResult(
            outputs=[
                {
                    "type": "error",
                    "message": "Gateway configuration error: PluginManager missing.",
                }
            ]
        )

    if not proxied_server:
        logger.error(f"Attempted to run tool on unknown server: {server_name}")
        return types.CallToolResult(
            outputs=[
                {
                    "type": "error",
                    "message": f"Proxied server '{server_name}' not found.",
                }
            ]
        )
    if proxied_server.session is None:
        logger.error(f"Attempted to run tool on inactive server: {server_name}")
        return types.CallToolResult(
            outputs=[
                {
                    "type": "error",
                    "message": f"Proxied server '{server_name}' session is not active.",
                }
            ]
        )

    logger.info(
        f"Routing tool call '{tool_name}' to server '{server_name}' via plugins"
    )
    try:
        # Pass plugin_manager and mcp_context down to the server's call_tool method
        # The Server.call_tool method now handles calling the sanitizers with the plugin manager
        result = await proxied_server.call_tool(
            plugin_manager=plugin_manager,
            name=tool_name,
            arguments=arguments,
            mcp_context=ctx,  # Pass the gateway's context if plugins need it
        )
        return result
    except SanitizationError as se:
        # Catch specific sanitization errors raised by plugins (via sanitizers.py)
        logger.error(
            f"Sanitization policy violation for tool '{tool_name}' on server '{server_name}': {se}"
        )
        # Return the error structured for the LLM
        return types.CallToolResult(
            outputs=[{"type": "error", "message": f"Gateway policy violation: {se}"}]
        )
    except Exception as e:
        # Catch other general exceptions during the proxied tool call or plugin execution
        logger.error(
            f"Error processing tool '{tool_name}' on server '{server_name}': {e}",
            exc_info=True,
        )
        return types.CallToolResult(
            outputs=[
                {
                    "type": "error",
                    "message": f"Error executing tool '{tool_name}' on server '{server_name}': {e}",
                }
            ]
        )


# --- Argument Parsing ---
def parse_args(args=None):
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="MCP Gateway Server")
    parser.add_argument(
        "--mcp-json-path",
        type=str,
        required=True,
        help="Path to the mcp.json configuration file",
    )
    parser.add_argument(
        "--enable-guardrails",
        action="append",
        help="Enable specific guardrail plugins (e.g., 'basic', 'lasso'). Multiple plugins can be enabled by repeating the argument. If used without a value, all guardrail plugins are enabled.",
        nargs="?",
        const="all",  # Default when flag is used without value
        default=[],  # Default when flag is not used at all
    )
    parser.add_argument(
        "--enable-tracing",
        action="append",
        help="Enable specific tracing plugins. Multiple plugins can be enabled by repeating the argument. If used without a value, all tracing plugins are enabled.",
        nargs="?",
        const="all",  # Default when flag is used without value
        default=[],  # Default when flag is not used at all
    )
    parser.add_argument(
        "--plugin-dir",
        type=str,
        default=None,
        help="Path to the directory containing plugin subdirectories (e.g., 'guardrails', 'tracing'). Defaults to './plugins' relative to server.py.",
    )
    # Add other arguments like host/port if needed for different transports later
    # parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    # parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    if args is None:
        args = sys.argv[1:]
    return parser.parse_args(args)


def main():
    # Parse args and store globally for lifespan access
    global cli_args
    cli_args = parse_args()

    logger.info("Starting MCP gateway server directly...")
    # mcp.run() defaults to stdio transport
    # If you need other transports (like HTTP SSE), configure them here:
    # Example for HTTP:
    # import uvicorn
    # uvicorn.run(mcp, host=cli_args.host, port=cli_args.port)
    mcp.run()


if __name__ == "__main__":
    main()
