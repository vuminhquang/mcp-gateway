import logging
import os
from typing import Any, Dict, List, Optional
from xetrack import Tracker
from xetrack.logging import LOGURU_PARAMS
from mcp_gateway.plugins.base import TracingPlugin, PluginContext

logger = logging.getLogger(__name__)

class XetrackParams:
    """Parameters for Xetrack tracing."""
    WARNINGS:bool = os.getenv("XETRACK_WARNINGS", "False").lower() == "true"
    LOG_SYSTEM_PARAMS:bool = os.getenv("XETRACK_LOG_SYSTEM_PARAMS", "false").lower() == "true"
    LOG_NETWORK_PARAMS:bool = os.getenv("XETRACK_LOG_NETWORK_PARAMS", "false").lower() == "true"    
    DB_PATH:str = os.getenv("XETRACK_DB_PATH", Tracker.SKIP_INSERT)
    LOGS_PATH:str|None = os.getenv("XETRACK_LOGS_PATH", None)
    LOGS_STDOUT:bool = os.getenv("XETRACK_LOGS_STDOUT", "false").lower() == "true"
    FLATTEN_RESPONSE:bool = os.getenv("XETRACK_FLATTEN_RESPONSE", "true").lower() == "true"
    FLATTEN_ARGUMENTS:bool = os.getenv("XETRACK_FLATTEN_ARGUMENTS", "true").lower() == "true"    
    LOG_FORMAT:str = os.getenv("XETRACK_LOG_FORMAT", LOGURU_PARAMS.LOG_FILE_FORMAT)

def to_events(context: PluginContext, event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Format the response data for logging.
    
    Args:
        context: The plugin context containing the response
        event: The event dictionary to update with response data
        
    Returns:
        The updated event dictionary
    """

    if XetrackParams.FLATTEN_ARGUMENTS:
        arguments = context.arguments or {}
        for key, value in arguments.items():
            event[key] = value
        event.pop("arguments", None)

    response = context.response or {}
    event["response_type"] = 'unknown' if response is None else type(response).__name__ # type: ignore
    for key, value in response.items():
        event[key] = value        
    event.pop("response", None)

    
    try:
        if hasattr(context.response, "model_dump"):
            response = context.response.model_dump()
        elif (
            isinstance(response, tuple)
            and len(response) == 2 # type: ignore
        ):
            # For resource responses (content, mime_type)
            content, mime_type = context.response
            if mime_type and ("text" in mime_type or "json" in mime_type):
                try:
                    content_str = content.decode("utf-8", errors="replace")                            
                    event["content"] = content_str
                    
                except:
                    event["content"] = "<binary data>"
            else:
                event["content"] = (
                    f"<binary data ({len(content)} bytes)>"
                )
            event["mime_type"] = mime_type
    except Exception as e:
        event["error_getting_response"] = str(e)    


    if not XetrackParams.FLATTEN_RESPONSE:
        return [event]
        
    events = []    
    for content in event.pop("content", []):
        content_event = event.copy()
        for key, value in content.items():
            content_event[f"content_{key}"] = value
        
        events.append(content_event)    
    

    return events

class XetrackTracingPlugin(TracingPlugin):
    """A plugin for Xetrack tracing."""

    plugin_type = "tracing"
    plugin_name = "xetrack"


    def __init__(self):
        self.db_path:str = XetrackParams.DB_PATH
        self.logs_path:str|None = XetrackParams.LOGS_PATH
        self.logs_stdout:bool = XetrackParams.LOGS_STDOUT
        self.tracker:Tracker|None = None
        

    def load(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Loads configuration for the tracing plugin.

        Configuration options:
        - db_path: The path to the database file (default: Tracker.IN_MEMORY)
        - logs_path: The path to the logs file (default: None)
        - logs_stdout: Whether to log to stdout (default: False)
        """
        if config is None:
            config = {}      
        self.logs_path = config.get("logs_path", self.logs_path)
        self.logs_stdout = config.get("logs_stdout", self.logs_stdout)
        self.db_path = config.get("db_path", self.db_path)
        self.tracker = Tracker(db=self.db_path, 
                               logs_path=self.logs_path, 
                               logs_stdout=self.logs_stdout, 
                               log_system_params=False,
                               log_network_params=False,
                               logs_file_format=XetrackParams.LOG_FORMAT,
                               warnings=XetrackParams.WARNINGS)
        logger.info(
            f"XetrackTracingPlugin loaded with db_path={self.db_path}, "
            f"logs_path={self.logs_path}, logs_stdout={self.logs_stdout}"
        )

    def process_request(self, context: PluginContext) -> Optional[Dict[str, Any]]:
        """Logs request data."""
        # Tracing plugins don't modify the request
        return context.arguments

    def process_response(self, context: PluginContext) -> Any:
        """Logs response data."""    
        event: Dict[str, Any] = context.to_dict()
        events = to_events(context, event)
        for event in events:
            self.tracker.log(event)
        return context.response
