"""
Tests for LLM retry logic and error handling.
Validates that call_ollama retries on failure and raises LLMError appropriately.
"""
import pytest
from unittest.mock import patch, MagicMock
from pydantic_agents import call_ollama, parse_json_response, LLMError


class TestCallOllamaRetry:
    """Test retry logic in call_ollama."""

    @patch("pydantic_agents.rag_cache")
    @patch("pydantic_agents._get_github_client")
    def test_succeeds_on_first_attempt(self, mock_client_fn, mock_cache):
        """Should return result immediately on success."""
        mock_cache.get.return_value = None  # No cache
        mock_cache.make_key.return_value = "test_key"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"result": "ok"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        result = call_ollama("system", "user", max_retries=3)
        assert result == '{"result": "ok"}'
        assert mock_client.chat.completions.create.call_count == 1

    @patch("pydantic_agents.rag_cache")
    @patch("pydantic_agents._get_github_client")
    def test_retries_on_failure_then_succeeds(self, mock_client_fn, mock_cache):
        """Should retry and succeed on second attempt."""
        mock_cache.get.return_value = None
        mock_cache.make_key.return_value = "test_key"

        mock_client = MagicMock()

        # First call fails, second succeeds
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"status": "recovered"}'

        mock_client.chat.completions.create.side_effect = [
            Exception("Connection timeout"),
            mock_response,
        ]
        mock_client_fn.return_value = mock_client

        result = call_ollama("system", "user", max_retries=3)
        assert "recovered" in result
        assert mock_client.chat.completions.create.call_count == 2

    @patch("pydantic_agents.rag_cache")
    @patch("pydantic_agents._get_github_client")
    def test_raises_after_all_retries_exhausted(self, mock_client_fn, mock_cache):
        """Should raise LLMError after max_retries failures."""
        mock_cache.get.return_value = None
        mock_cache.make_key.return_value = "test_key"

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_client_fn.return_value = mock_client

        with pytest.raises(LLMError, match="failed after 2 attempts"):
            call_ollama("system", "user", max_retries=2)

        assert mock_client.chat.completions.create.call_count == 2

    @patch("pydantic_agents.rag_cache")
    @patch("pydantic_agents._get_github_client")
    def test_empty_response_raises_immediately(self, mock_client_fn, mock_cache):
        """Empty LLM response should raise without retrying."""
        mock_cache.get.return_value = None
        mock_cache.make_key.return_value = "test_key"

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        mock_client_fn.return_value = mock_client

        with pytest.raises(LLMError, match="empty response"):
            call_ollama("system", "user", max_retries=3)


class TestParseJsonResponse:
    """Test JSON parsing from LLM responses."""

    def test_parses_clean_json(self):
        """Should parse clean JSON."""
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_in_markdown_block(self):
        """Should extract JSON from markdown code blocks."""
        response = '```json\n{"risk_level": "HIGH", "risk_score": 75}\n```'
        result = parse_json_response(response)
        assert result["risk_level"] == "HIGH"
        assert result["risk_score"] == 75

    def test_parses_json_with_surrounding_text(self):
        """Should find JSON object in text with preamble."""
        response = 'Here is my analysis:\n{"risk_level": "LOW", "risk_score": 25}\nEnd.'
        result = parse_json_response(response)
        assert result["risk_level"] == "LOW"

    def test_raises_on_invalid_json(self):
        """Should raise ValueError on unparseable content."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_json_response("This is not JSON at all")

    def test_raises_on_empty_string(self):
        """Should raise ValueError on empty input."""
        with pytest.raises(ValueError):
            parse_json_response("")
