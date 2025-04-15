# MCP-Proxy Tests

This directory contains tests for the MCP-Proxy project.

## Running Tests

To run all tests:

```bash
python -m pytest
```

To run a specific test file:

```bash
python -m pytest tests/test_lasso_guardrail.py
```

## Lasso Guardrail Tests

The Lasso guardrail tests require a valid Lasso API key. You can set this up in one of two ways:

1. Create a `.env` file in the project root with the following content:
   ```
   LASSO_API_KEY=your_api_key_here
   LASSO_USER_ID=optional_user_id
   LASSO_CONVERSATION_ID=optional_conversation_id
   ```

2. Set the environment variables directly:
   ```bash
   export LASSO_API_KEY=your_api_key_here
   ```

Tests that require the API key are skipped automatically if no key is provided.

## Test Structure

- `test_basic_guardrail.py`: Basic tests for guardrail functionality
- `test_lasso_guardrail.py`: Tests for the Lasso Security API integration
- `simple_pii_example.py`: Example script demonstrating PII detection

## Adding New Tests

When adding new tests, please follow these guidelines:

1. Use proper type annotations and docstrings
2. Properly configure logging using the standard logging module
3. Use pytest fixtures and markers appropriately
4. Follow the existing patterns for interacting with the API
5. Skip tests appropriately when external dependencies are not available 