import importlib
import inspect
import logging
import os
import pkgutil
from typing import Any, Dict, List, Optional, Type

from mcp_gateway.plugins.base import (
    Plugin,
    PluginContext,
    GuardrailPlugin,
    TracingPlugin,
)

logger = logging.getLogger(__name__)


class PluginManager:
    """Discovers, loads, and manages plugins."""

    def __init__(self, plugin_dirs: List[str]):
        """Initializes the PluginManager.

        Args:
            plugin_dirs: List of directories to search for plugins.
        """
        self.plugin_dirs = plugin_dirs
        self._plugins: Dict[str, List[Plugin]] = {
            GuardrailPlugin.plugin_type: [],
            TracingPlugin.plugin_type: [],
        }
        self._loaded_plugin_types: Dict[Type[Plugin], Plugin] = {}

    def discover_and_load(
        self,
        enabled_types: Optional[List[str]] = None,
        plugin_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        enabled_plugins: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """Discovers plugins in specified directories and loads enabled ones.

        Args:
            enabled_types: List of plugin types to enable (e.g., ['guardrail', 'tracing']).
                           If None, all discovered types are potentially loadable based on config.
            plugin_configs: Optional configuration dictionary for specific plugins,
                            keyed by plugin class name.
            enabled_plugins: Optional dictionary mapping plugin types to lists of plugin names
                            to enable (e.g., {'guardrail': ['basic', 'lasso']}).
                            If a type has an empty list or contains 'all', all plugins of that type are enabled.
        """
        logger.info(f"Discovering plugins in: {self.plugin_dirs}")
        if plugin_configs is None:
            plugin_configs = {}
        if enabled_plugins is None:
            enabled_plugins = {}

        # Only discover and load plugins for enabled types
        if enabled_types is None or not enabled_types:
            logger.info("No plugin types enabled, skipping plugin discovery.")
            return

        plugin_classes = self._discover_plugins_lazily(enabled_types, enabled_plugins)
        logger.info(
            f"Discovered {len(plugin_classes)} potential plugin classes to load."
        )

        for plugin_cls in plugin_classes:
            plugin_type = getattr(plugin_cls, "plugin_type", "base")
            plugin_name = plugin_cls.__name__

            # Use plugin_name attribute if available, otherwise normalize the class name
            custom_plugin_name = getattr(plugin_cls, "plugin_name", "")
            if custom_plugin_name:
                normalized_name = custom_plugin_name.lower()
                logger.debug(
                    f"Using explicit plugin_name '{normalized_name}' for {plugin_name}"
                )
            else:
                # Legacy normalization for backward compatibility
                normalized_name = plugin_name.lower()
                for type_name in ["guardrail", "tracing", "plugin"]:
                    normalized_name = normalized_name.replace(type_name, "")
                logger.debug(
                    f"Normalized plugin {plugin_name} to '{normalized_name}' (no explicit plugin_name)"
                )

            # Load configuration for this specific plugin
            config = plugin_configs.get(plugin_name, {})

            try:
                plugin_instance = plugin_cls()
                plugin_instance.load(config)
                if plugin_type in self._plugins:
                    self._plugins[plugin_type].append(plugin_instance)
                    self._loaded_plugin_types[plugin_cls] = plugin_instance
                    logger.info(
                        f"Successfully loaded and configured plugin: {plugin_name} (type: {plugin_type})"
                    )
                else:
                    logger.warning(
                        f"Plugin {plugin_name} has unknown type '{plugin_type}'. Skipping."
                    )

            except Exception as e:
                logger.error(
                    f"Failed to load or configure plugin {plugin_name}: {e}",
                    exc_info=True,
                )

        for p_type, p_list in self._plugins.items():
            logger.info(f"Loaded {len(p_list)} plugins of type '{p_type}'")

    def _discover_plugins_lazily(
        self, enabled_types: List[str], enabled_plugins: Dict[str, List[str]]
    ) -> List[Type[Plugin]]:
        """
        Discovers plugin classes within the specified directories, but only imports
        modules for plugins that are explicitly enabled.
        """
        plugin_classes = []

        # Create a mapping of known plugin types to directories
        # This handles singular/plural differences
        type_to_dir_mapping = {
            "guardrail": ["guardrail", "guardrails"],
            "tracing": ["tracing", "tracings"],
        }

        for plugin_dir_path in self.plugin_dirs:
            if not os.path.isdir(plugin_dir_path):
                logger.warning(f"Plugin directory not found: {plugin_dir_path}")
                continue

            # Determine proper module path
            pkg_path = "mcp_gateway.plugins"
            if "mcp_gateway/plugins" in plugin_dir_path:
                subpath = plugin_dir_path.split("mcp_gateway/plugins/")[-1]
                module_base_path = f"{pkg_path}.{subpath.replace('/', '.')}"
            else:
                dir_name = os.path.basename(plugin_dir_path)
                module_base_path = f"{pkg_path}.{dir_name}"

            # Get directory name to map to a plugin type
            dir_name = os.path.basename(plugin_dir_path).lower()

            # Find the matching plugin type for this directory
            matched_type = None
            for plugin_type, dir_variants in type_to_dir_mapping.items():
                if dir_name in dir_variants:
                    matched_type = plugin_type
                    break

            if matched_type is None:
                logger.warning(
                    f"Could not determine plugin type from directory: {dir_name}"
                )
                continue

            # Skip if this plugin type is not enabled
            if matched_type not in enabled_types:
                logger.debug(
                    f"Skipping plugin directory {plugin_dir_path} - type {matched_type} not enabled"
                )
                continue

            logger.info(
                f"Searching for modules in: {plugin_dir_path} (as {module_base_path})"
            )

            # Get the list of enabled plugin names for this type
            enabled_names = enabled_plugins.get(matched_type, [])

            # If 'all' is in the list or the list is empty, we'll try to load all plugins
            load_all_of_type = not enabled_names or "all" in enabled_names

            for finder, name, ispkg in pkgutil.iter_modules([plugin_dir_path]):
                # If we're not loading all plugins of this type, check if this plugin is explicitly enabled
                if not load_all_of_type:
                    # Normalize module name for comparison
                    normalized_module = name.lower()

                    # Skip if this specific plugin is not in the enabled list
                    if normalized_module not in [p.lower() for p in enabled_names]:
                        logger.debug(
                            f"Skipping plugin module {name} - not explicitly enabled"
                        )
                        continue

                full_module_name = f"{module_base_path}.{name}"
                try:
                    logger.info(f"Importing module: {full_module_name}")
                    module = importlib.import_module(full_module_name)

                    for obj_name, obj in inspect.getmembers(module, inspect.isclass):
                        # Check if it's a concrete subclass of Plugin but not Plugin itself
                        if (
                            issubclass(obj, Plugin)
                            and obj is not Plugin
                            and not inspect.isabstract(obj)
                        ):
                            logger.info(
                                f"Found plugin class: {obj.__name__} in {full_module_name}"
                            )
                            plugin_classes.append(obj)
                except ImportError as e:
                    logger.error(f"Failed to import module {full_module_name}: {e}")
                except Exception as e:
                    logger.error(
                        f"Error inspecting module {full_module_name}: {e}",
                        exc_info=True,
                    )

        return plugin_classes

    def get_plugins(self, plugin_type: str) -> List[Plugin]:
        """Returns loaded plugins of a specific type."""
        return self._plugins.get(plugin_type, [])

    async def run_request_plugins(
        self, context: PluginContext
    ) -> Optional[Dict[str, Any]]:
        """Runs all relevant plugins for a request, modifying arguments sequentially.

        Returns:
            The final arguments dictionary after all plugins, or None if blocked.
        """
        current_args = context.arguments

        # Run Tracing first (doesn't modify args by default)
        for plugin in self.get_plugins(TracingPlugin.plugin_type):
            try:
                _ = plugin.process_request(
                    context._replace(arguments=current_args)
                )  # Pass current args
            except Exception as e:
                logger.error(
                    f"Error running tracing request plugin {plugin.__class__.__name__}: {e}",
                    exc_info=True,
                )

        # Run Guardrails (can modify args)
        for plugin in self.get_plugins(GuardrailPlugin.plugin_type):
            if current_args is None:  # If a previous guardrail blocked
                break
            try:
                context_for_plugin = PluginContext(
                    server_name=context.server_name,
                    capability_type=context.capability_type,
                    capability_name=context.capability_name,
                    arguments=current_args,  # Pass potentially modified args
                    mcp_context=context.mcp_context,
                )
                # Check if process_request is an awaitable coroutine
                if inspect.iscoroutinefunction(plugin.process_request):
                    current_args = await plugin.process_request(context_for_plugin)
                else:
                    current_args = plugin.process_request(context_for_plugin)
            except Exception as e:
                logger.error(
                    f"Error running guardrail request plugin {plugin.__class__.__name__}: {e}",
                    exc_info=True,
                )
                # Optionally block on error, or just log and continue
                # current_args = None # Block if any plugin fails
                # break

        return current_args

    async def run_response_plugins(self, context: PluginContext) -> Any:
        """Runs all relevant plugins for a response, modifying the response sequentially.

        Returns:
            The final response after all plugins.
        """
        current_response = context.response

        # Run Guardrails first (can modify response)
        for plugin in self.get_plugins(GuardrailPlugin.plugin_type):
            try:
                context_for_plugin = PluginContext(
                    server_name=context.server_name,
                    capability_type=context.capability_type,
                    capability_name=context.capability_name,
                    arguments=context.arguments,  # Original arguments
                    response=current_response,  # Pass potentially modified response
                    mcp_context=context.mcp_context,
                )
                # Check if process_response is an awaitable coroutine
                if inspect.iscoroutinefunction(plugin.process_response):
                    current_response = await plugin.process_response(context_for_plugin)
                else:
                    current_response = plugin.process_response(context_for_plugin)
            except Exception as e:
                logger.error(
                    f"Error running guardrail response plugin {plugin.__class__.__name__}: {e}",
                    exc_info=True,
                )
                # Handle error, maybe return an error object or raise?
                # For now, just log and continue with potentially unmodified response

        # Run Tracing last (doesn't modify response by default)
        for plugin in self.get_plugins(TracingPlugin.plugin_type):
            try:
                # Create context with the latest response for tracing
                context_for_plugin = PluginContext(
                    server_name=context.server_name,
                    capability_type=context.capability_type,
                    capability_name=context.capability_name,
                    arguments=context.arguments,
                    response=current_response,  # Trace the final response
                    mcp_context=context.mcp_context,
                )
                # Check if process_response is an awaitable coroutine
                if inspect.iscoroutinefunction(plugin.process_response):
                    _ = await plugin.process_response(context_for_plugin)
                else:
                    _ = plugin.process_response(context_for_plugin)
            except Exception as e:
                logger.error(
                    f"Error running tracing response plugin {plugin.__class__.__name__}: {e}",
                    exc_info=True,
                )

        return current_response
