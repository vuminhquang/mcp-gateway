import logging
from typing import Any, Dict, Optional

from mcp_gateway.plugins.base import TracingPlugin, PluginContext

logger = logging.getLogger(__name__)


class BasicTracingPlugin(TracingPlugin):
    """A basic tracing plugin that logs request and response data."""

    plugin_type = "tracing"
    plugin_name = "basic"

    def __init__(self):
        self.log_level = logging.INFO
        self.log_detailed_content = False

    def load(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Loads configuration for the tracing plugin.

        Configuration options:
        - log_level: The logging level (default: INFO)
        - log_detailed_content: Whether to log detailed content or just metadata (default: False)
        """
        if config is None:
            config = {}

        # Get log level from config with fallback to INFO
        log_level_name = config.get("log_level", "INFO")
        try:
            self.log_level = getattr(logging, log_level_name)
        except (AttributeError, TypeError):
            logger.warning(f"Invalid log level: {log_level_name}, using INFO instead")
            self.log_level = logging.INFO

        # Whether to log detailed content
        self.log_detailed_content = config.get("log_detailed_content", False)

        logger.info(
            f"BasicTracingPlugin loaded with log_level={log_level_name}, "
            f"log_detailed_content={self.log_detailed_content}"
        )

    def process_request(self, context: PluginContext) -> Optional[Dict[str, Any]]:
        """Logs request data."""
        request_info = {
            "server": context.server_name,
            "type": context.capability_type,
            "name": context.capability_name,
        }

        # Log arguments if detailed content is enabled
        if self.log_detailed_content and context.arguments:
            request_info["arguments"] = context.arguments

        logger.log(self.log_level, f"Tracing request: {request_info}")

        # Tracing plugins don't modify the request
        return context.arguments

    def process_response(self, context: PluginContext) -> Any:
        """Logs response data."""
        response_info = {
            "server": context.server_name,
            "type": context.capability_type,
            "name": context.capability_name,
        }

        # Add basic response type information
        if context.response is not None:
            response_info["response_type"] = type(context.response).__name__

            # Log detailed response content if enabled
            if self.log_detailed_content:
                # Try to get a reasonable string representation of the response
                try:
                    if hasattr(context.response, "model_dump"):
                        response_info["response"] = context.response.model_dump()
                    elif (
                        isinstance(context.response, tuple)
                        and len(context.response) == 2
                    ):
                        # For resource responses (content, mime_type)
                        content, mime_type = context.response
                        if mime_type and ("text" in mime_type or "json" in mime_type):
                            try:
                                content_str = content.decode("utf-8", errors="replace")
                                # Truncate if too long
                                if len(content_str) > 1000:
                                    content_str = content_str[:1000] + "... [truncated]"
                                response_info["content"] = content_str
                            except:
                                response_info["content"] = "<binary data>"
                        else:
                            response_info["content"] = (
                                f"<binary data ({len(content)} bytes)>"
                            )
                        response_info["mime_type"] = mime_type
                except Exception as e:
                    response_info["error_getting_response"] = str(e)

        logger.log(self.log_level, f"Tracing response: {response_info}")

        # Tracing plugins don't modify the response
        return context.response
