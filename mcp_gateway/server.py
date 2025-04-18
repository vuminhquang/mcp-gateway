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
        self._server_info: Optional[types.InitializeResult] = None
        logger.info(f"Initialized Proxied Server object: {self.name}")

    @property
    def session(self) -> ClientSession:
        """Returns the active ClientSession, raising an error if not started."""
        if self._session is None:
            # This should ideally not happen if manage_lifecycle is used correctly
            raise RuntimeError(f"Server '{self.name}' session not available or lifecycle not managed.")
        return self._session

    @asynccontextmanager
    async def manage_lifecycle(self) -> AsyncIterator[None]:
        """Async context manager to handle the startup and shutdown of the server's session."""
        logger.info(f"[{self.name}] Entering manage_lifecycle context...")
        local_exit_stack = AsyncExitStack() # Stack specific to this server's lifecycle
        try:
            logger.info(f"[{self.name}] Starting underlying process and establishing connection...")
            server_params = StdioServerParameters(
                command=self.config.get("command", ""),
                args=self.config.get("args", []),
                env=self.config.get("env", None),
                cwd=self.config.get("cwd", None) # Add cwd if needed
            )

            # Enter stdio_client context managed by the local stack
            stdio_cm = stdio_client(server_params)
            read, write = await local_exit_stack.enter_async_context(stdio_cm)

            # Enter ClientSession context managed by the local stack
            session_cm = ClientSession(read, write)
            self._session = await local_exit_stack.enter_async_context(session_cm)

            # Initialize the session
            self._server_info = await self._session.initialize()
            logger.info(f"[{self.name}] Started and initialized successfully within manage_lifecycle.")

            yield # Server is ready and session is active

        except Exception as e:
            logger.error(f"[{self.name}] Failed during manage_lifecycle startup: {e}", exc_info=True)
            # Ensure members are reset even if startup fails partially
            self._session = None
            self._server_info = None
            # Allow the exception to propagate so lifespan knows it failed
            raise 
        finally:
            logger.info(f"[{self.name}] Exiting manage_lifecycle context, cleaning up resources...")
            # The local_exit_stack will automatically clean up stdio_client and ClientSession
            await local_exit_stack.aclose() 
            # Reset members after cleanup
            self._session = None
            self._server_info = None
            logger.info(f"[{self.name}] Resources cleaned up.")

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
    
    # Use a single ExitStack for the entire lifespan to manage servers
    async with AsyncExitStack() as lifespan_stack:
        if not proxied_server_configs:
            logger.warning(
                "No proxied MCP servers configured. Running in standalone mode (plugins still active)."
            )
        else:
            logger.info(f"Attempting to start {len(proxied_server_configs)} configured proxied servers...")
            successfully_started_servers = {}
            for name, server_config in proxied_server_configs.items():
                logger.info(f"Creating and starting lifecycle management for proxied server: {name}")
                proxied_server = Server(name, server_config)
                try:
                    # Enter the lifecycle context for this server using the lifespan stack
                    await lifespan_stack.enter_async_context(proxied_server.manage_lifecycle())
                    # If successful, add it to the context for tools
                    successfully_started_servers[name] = proxied_server
                    logger.info(f"Successfully started and added server '{name}' to context.")
                except Exception as e:
                    # Log error but continue trying to start other servers
                    logger.error(f"Failed to start server '{name}' within lifespan: {e}", exc_info=False) # Don't need full trace usually
            
            context.proxied_servers = successfully_started_servers # Only add successfully started servers
            logger.info(f"Finished starting servers. {len(context.proxied_servers)} successfully started.")

        # Yield the context with successfully started servers
        logger.info("MCP Gateway Lifespan entering operational state...")
        yield context
        logger.info("MCP Gateway Lifespan exiting operational state...")

    # --- Cleanup --- 
    # The lifespan_stack automatically handles calling __aexit__ on each 
    # successfully entered server's manage_lifecycle context upon exiting the 'async with'.
    logger.info("MCP gateway lifespan shutdown complete (handled by AsyncExitStack).")


# Initialize the MCP gateway server
# Pass description and version if desired
mcp = FastMCP("MCP Gateway", lifespan=lifespan, version="0.1.0")


@mcp.tool()
async def get_metadata(ctx: Context) -> Dict[str, Any]:
    """Internal tool to retrieve metadata about all proxied servers and their tools."""
    gateway_context: GetewayContext = ctx.request_context.lifespan_context
    all_metadata = {}
    logger.info("[Gateway Tool] Received request for get_metadata")

    if not gateway_context or not gateway_context.proxied_servers:
        logger.warning("[Gateway Tool] get_metadata called, but no proxied servers found in context.")
        return {"error": "No proxied servers configured or available."}

    for server_name, server_instance in gateway_context.proxied_servers.items():
        logger.info(f"[Gateway Tool] Processing get_metadata for proxied server: {server_name}")
        server_meta = {"description": server_instance.config.get("description", "N/A"), "tools": []}
        try:
            # Ensure the server session is started and initialized before querying
            if server_instance._session is None or server_instance._server_info is None:
                 logger.warning(f"[Gateway Tool] Proxied server '{server_name}' session/info not ready. Attempting start...")
                 # This might be risky if called concurrently, but let's try
                 # await server_instance.start() # Avoid starting here, should be done in lifespan
                 if server_instance._session is None or server_instance._server_info is None:
                      logger.error(f"[Gateway Tool] Proxied server '{server_name}' still not ready after check. Skipping.")
                      server_meta["error"] = "Server not initialized"
                      all_metadata[server_name] = server_meta
                      continue

            logger.debug(f"[Gateway Tool] Calling list_tools on proxied server: {server_name}")
            tools_list = await server_instance.list_tools()
            logger.debug(f"[Gateway Tool] list_tools returned for {server_name}. Found {len(tools_list)} tools.")

            if tools_list:
                 # Convert tools to the expected dictionary format if necessary
                 # Assuming list_tools returns a list of types.Tool objects
                 for tool in tools_list:
                     tool_dict = {
                         "name": tool.name,
                         "description": tool.description,
                         # Use model_dump or dict() depending on pydantic version if inputSchema is a model
                         "inputSchema": tool.inputSchema.model_dump() if hasattr(tool.inputSchema, 'model_dump') else tool.inputSchema.dict() if hasattr(tool.inputSchema, 'dict') else tool.inputSchema,
                         # Add other relevant fields if needed
                     }
                     server_meta["tools"].append(tool_dict)
            all_metadata[server_name] = server_meta
        except Exception as e:
            logger.error(f"[Gateway Tool] Error getting metadata for server '{server_name}': {e}", exc_info=True)
            server_meta["error"] = str(e)
            all_metadata[server_name] = server_meta
    
    logger.info("[Gateway Tool] Finished processing get_metadata for all servers.")
    return all_metadata


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
