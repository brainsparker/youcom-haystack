# SPDX-FileCopyrightText: 2026-present You.com <developers@you.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret
from requests.exceptions import HTTPError

from haystack_integrations.components.websearch.youcom import YouComWebSearch
from haystack_integrations.components.websearch.youcom.youcom_websearch import (
    USER_AGENT,
    YOUCOM_KEYED_SEARCH_URL,
    YOUCOM_KEYLESS_SEARCH_URL,
)

SAMPLE_RESPONSE = {
    "results": {
        "web": [
            {
                "url": "https://haystack.deepset.ai",
                "title": "Haystack | Haystack",
                "description": "Open-source AI framework",
                "snippets": ["Haystack is an open-source framework", "for building production-ready AI pipelines"],
                "page_age": "2026-07-01T00:00:00",
            },
        ],
        "news": [
            {
                "url": "https://example.com/news",
                "title": "AI news item",
                "description": "A news description",
                "page_age": "2026-07-20T00:00:00",
            },
        ],
    },
    "metadata": {"query": "test", "latency": 0.1},
}


def mock_response(json_body, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = json_body
    return response


class TestYouComWebSearch:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)

    def test_init_defaults(self):
        component = YouComWebSearch()
        assert component.api_key == Secret.from_env_var("YDC_API_KEY", strict=False)
        assert component.top_k == 10
        assert component.freshness is None
        assert component.country is None
        assert component.search_lang is None
        assert component.safesearch is None
        assert component.extra_params is None
        assert component.timeout == 10
        assert component.max_retries == 3

    def test_init_without_env_var_does_not_raise(self):
        component = YouComWebSearch()
        assert component.api_key.resolve_value() is None

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_keyless_when_no_api_key(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["url"] == YOUCOM_KEYLESS_SEARCH_URL
        assert "X-API-Key" not in kwargs["headers"]

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_keyed_when_env_var_set(self, mock_request, monkeypatch):
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["url"] == YOUCOM_KEYED_SEARCH_URL
        assert kwargs["headers"]["X-API-Key"] == "env-key"

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_keyed_with_explicit_token(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(api_key=Secret.from_token("explicit-key"))
        component.run(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["url"] == YOUCOM_KEYED_SEARCH_URL
        assert kwargs["headers"]["X-API-Key"] == "explicit-key"

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_sends_user_agent(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        component.run(query="test")

        headers = mock_request.call_args.kwargs["headers"]
        assert headers["User-Agent"] == USER_AGENT
        assert USER_AGENT.startswith("youcom-haystack/")

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_builds_params(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(
            top_k=5,
            freshness="week",
            country="US",
            search_lang="EN",
            safesearch="moderate",
            extra_params={"include_domains": "nytimes.com,bbc.com"},
        )
        component.run(query="climate news")

        params = mock_request.call_args.kwargs["params"]
        assert params == {
            "query": "climate news",
            "count": 5,
            "freshness": "week",
            "country": "US",
            "language": "EN",
            "safesearch": "moderate",
            "include_domains": "nytimes.com,bbc.com",
        }

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_top_k_override(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(top_k=10)
        component.run(query="q", top_k=3)

        assert mock_request.call_args.kwargs["params"]["count"] == 3

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_parses_documents_and_links(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch()
        result = component.run(query="test")

        documents = result["documents"]
        links = result["links"]
        assert len(documents) == 2
        assert all(isinstance(doc, Document) for doc in documents)

        web_doc = documents[0]
        assert web_doc.content == "Haystack is an open-source framework\nfor building production-ready AI pipelines"
        assert web_doc.meta == {
            "title": "Haystack | Haystack",
            "url": "https://haystack.deepset.ai",
            "source": "web",
            "page_age": "2026-07-01T00:00:00",
        }

        news_doc = documents[1]
        assert news_doc.content == "A news description"
        assert news_doc.meta["source"] == "news"

        assert links == ["https://haystack.deepset.ai", "https://example.com/news"]

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_empty_results(self, mock_request):
        mock_request.return_value = mock_response({"results": {"web": [], "news": []}})

        component = YouComWebSearch()
        result = component.run(query="test")

        assert result == {"documents": [], "links": []}

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_run_missing_sections(self, mock_request):
        mock_request.return_value = mock_response({"metadata": {}})

        component = YouComWebSearch()
        result = component.run(query="test")

        assert result == {"documents": [], "links": []}

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_keyless_rate_limit_error_includes_upgrade_hint(self, mock_request):
        error_response = MagicMock()
        error_response.status_code = 402
        mock_request.side_effect = HTTPError("402 Client Error", response=error_response)

        component = YouComWebSearch()
        with pytest.raises(HTTPError, match="YDC_API_KEY"):
            component.run(query="test")

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_keyed_error_is_not_rewritten(self, mock_request):
        error_response = MagicMock()
        error_response.status_code = 401
        mock_request.side_effect = HTTPError("401 Client Error", response=error_response)

        component = YouComWebSearch(api_key=Secret.from_token("bad-key"))
        with pytest.raises(HTTPError, match="401") as excinfo:
            component.run(query="test")
        assert "YDC_API_KEY" not in str(excinfo.value)

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.async_request_with_retry")
    async def test_run_async(self, mock_request):
        mock_request_async = AsyncMock(return_value=mock_response(SAMPLE_RESPONSE))
        mock_request.side_effect = mock_request_async

        component = YouComWebSearch()
        result = await component.run_async(query="test")

        assert len(result["documents"]) == 2
        assert mock_request.call_args.kwargs["url"] == YOUCOM_KEYLESS_SEARCH_URL

    def test_serialization_roundtrip_without_env_var(self):
        component = YouComWebSearch(top_k=7, freshness="month")
        data = component_to_dict(component, name="websearch")
        restored = component_from_dict(YouComWebSearch, data, name="websearch")

        assert restored.top_k == 7
        assert restored.freshness == "month"
        assert restored.api_key == Secret.from_env_var("YDC_API_KEY", strict=False)

    def test_serialization_roundtrip_with_env_var(self, monkeypatch):
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        component = YouComWebSearch()
        data = component_to_dict(component, name="websearch")
        restored = component_from_dict(YouComWebSearch, data, name="websearch")

        assert restored.api_key.resolve_value() == "env-key"

    def test_serialization_does_not_leak_token_secret(self):
        component = YouComWebSearch(api_key=Secret.from_token("super-secret"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component_to_dict(component, name="websearch")


@pytest.mark.integration
class TestYouComWebSearchIntegration:
    def test_keyless_live_search(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        component = YouComWebSearch(top_k=2)
        result = component.run(query="What is Haystack by deepset?")

        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert result["documents"][0].content

    @pytest.mark.skipif("YDC_API_KEY" not in os.environ, reason="YDC_API_KEY not set")
    def test_keyed_live_search(self):
        component = YouComWebSearch(top_k=2)
        result = component.run(query="What is Haystack by deepset?")

        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0

    @pytest.mark.skipif("YDC_API_KEY" not in os.environ, reason="YDC_API_KEY not set")
    async def test_keyed_live_search_async(self):
        component = YouComWebSearch(top_k=2)
        result = await component.run_async(query="What is Haystack by deepset?")

        assert len(result["documents"]) > 0
