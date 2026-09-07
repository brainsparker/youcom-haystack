# SPDX-FileCopyrightText: 2026-present You.com <developers@you.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from haystack import Document
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.tools import ComponentTool
from haystack.utils import Secret
from requests.exceptions import HTTPError

from haystack_integrations.components.websearch.youcom import YouComWebSearch, documents_to_text
from haystack_integrations.components.websearch.youcom.youcom_websearch import (
    DEFAULT_TOOL_DESCRIPTION,
    DEFAULT_TOOL_NAME,
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

HIGHLIGHTS_RESPONSE = {
    "results": {
        "web": [
            {
                "url": "https://haystack.deepset.ai",
                "title": "Haystack | Haystack",
                "description": "Open-source AI framework",
                "contents": {"highlights": ["Haystack 3 ships an Agent", "with tools and hooks"]},
            },
            {
                "url": "https://example.com/no-highlights",
                "title": "Fallback page",
                "description": "Only a description",
                "snippets": ["A snippet survives"],
                "contents": {"highlights": []},
            },
        ],
        "news": [],
    },
}

FULL_PAGE_RESPONSE = {
    "results": {
        "web": [
            {
                "url": "https://haystack.deepset.ai",
                "title": "Haystack | Haystack",
                "snippets": ["short fragment"],
                "contents": {"markdown": "# Haystack\n\nThe whole page in Markdown."},
            },
            {
                "url": "https://example.com/html-only",
                "title": "HTML only",
                "snippets": ["short fragment"],
                "contents": {"html": "<h1>HTML only</h1>"},
            },
            {
                "url": "https://example.com/crawl-failed",
                "title": "Crawl failed",
                "snippets": ["the snippet is all we have"],
                "contents": {},
            },
        ],
        "news": [],
    },
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
        assert component.extraction_mode == "snippets"
        assert component.extraction_formats is None
        assert component.crawl_timeout is None

    def test_init_rejects_unknown_extraction_mode(self):
        with pytest.raises(ValueError, match="extraction_mode"):
            YouComWebSearch(extraction_mode="everything")

    def test_init_rejects_unknown_extraction_format(self):
        with pytest.raises(ValueError, match="extraction_formats"):
            YouComWebSearch(extraction_mode="full_page", extraction_formats=["pdf"])
        with pytest.raises(ValueError, match="extraction_formats"):
            YouComWebSearch(extraction_mode="full_page", extraction_formats=[])

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
        assert kwargs["method"] == "GET"
        assert "json" not in kwargs
        assert kwargs["timeout"] == 10
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

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_snippets_mode_meta_is_unchanged(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(extraction_mode="snippets")
        result = component.run(query="test")

        assert "content_level" not in result["documents"][0].meta

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_highlights_mode_posts_extraction_body(self, mock_request):
        mock_request.return_value = mock_response(HIGHLIGHTS_RESPONSE)

        component = YouComWebSearch(
            api_key=Secret.from_token("key"),
            top_k=4,
            freshness="week",
            extraction_mode="highlights",
            extra_params={"include_domains": "deepset.ai, haystack.deepset.ai"},
        )
        component.run(query="haystack agent tools")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["url"] == YOUCOM_KEYED_SEARCH_URL
        assert kwargs["headers"]["X-API-Key"] == "key"
        assert "params" not in kwargs
        assert kwargs["json"] == {
            "query": "haystack agent tools",
            "count": 4,
            "freshness": "week",
            "include_domains": ["deepset.ai", "haystack.deepset.ai"],
            "extraction": {"extraction_mode": "highlights"},
        }

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_highlights_mode_parses_highlights_with_snippet_fallback(self, mock_request):
        mock_request.return_value = mock_response(HIGHLIGHTS_RESPONSE)

        component = YouComWebSearch(api_key=Secret.from_token("key"), extraction_mode="highlights")
        documents = component.run(query="test")["documents"]

        assert documents[0].content == "Haystack 3 ships an Agent\nwith tools and hooks"
        assert documents[0].meta["content_level"] == "highlights"
        assert "content_format" not in documents[0].meta
        assert documents[1].content == "A snippet survives"
        assert documents[1].meta["content_level"] == "snippets"

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_full_page_mode_posts_formats_and_crawl_timeout(self, mock_request):
        mock_request.return_value = mock_response(FULL_PAGE_RESPONSE)

        component = YouComWebSearch(
            api_key=Secret.from_token("key"),
            extraction_mode="full_page",
            extraction_formats=["markdown", "html"],
            crawl_timeout=30,
        )
        component.run(query="test", top_k=3)

        body = mock_request.call_args.kwargs["json"]
        assert body["count"] == 3
        assert body["extraction"] == {
            "extraction_mode": "full_page",
            "full_page": {"extraction_formats": ["markdown", "html"]},
        }
        assert body["crawl_timeout"] == 30

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_full_page_mode_omits_formats_when_not_set(self, mock_request):
        mock_request.return_value = mock_response(FULL_PAGE_RESPONSE)

        component = YouComWebSearch(api_key=Secret.from_token("key"), extraction_mode="full_page")
        component.run(query="test")

        body = mock_request.call_args.kwargs["json"]
        assert body["extraction"] == {"extraction_mode": "full_page"}
        assert "crawl_timeout" not in body

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_full_page_mode_prefers_markdown_then_html_then_snippets(self, mock_request):
        mock_request.return_value = mock_response(FULL_PAGE_RESPONSE)

        component = YouComWebSearch(api_key=Secret.from_token("key"), extraction_mode="full_page")
        documents = component.run(query="test")["documents"]

        assert documents[0].content == "# Haystack\n\nThe whole page in Markdown."
        assert documents[0].meta["content_level"] == "full_page"
        assert documents[0].meta["content_format"] == "markdown"

        assert documents[1].content == "<h1>HTML only</h1>"
        assert documents[1].meta["content_level"] == "full_page"
        assert documents[1].meta["content_format"] == "html"

        assert documents[2].content == "the snippet is all we have"
        assert documents[2].meta["content_level"] == "snippets"
        assert "content_format" not in documents[2].meta

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_extraction_requires_api_key(self, mock_request):
        component = YouComWebSearch(extraction_mode="highlights")

        with pytest.raises(ValueError, match="YDC_API_KEY"):
            component.run(query="test")
        mock_request.assert_not_called()

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.async_request_with_retry")
    async def test_run_async_highlights_mode(self, mock_request):
        mock_request.side_effect = AsyncMock(return_value=mock_response(HIGHLIGHTS_RESPONSE))

        component = YouComWebSearch(api_key=Secret.from_token("key"), extraction_mode="highlights")
        result = await component.run_async(query="test")

        kwargs = mock_request.call_args.kwargs
        assert kwargs["method"] == "POST"
        assert kwargs["json"]["extraction"] == {"extraction_mode": "highlights"}
        assert result["documents"][0].meta["content_level"] == "highlights"

    def test_documents_to_text(self):
        documents = [
            Document(content="First body", meta={"title": "First", "url": "https://one.example"}),
            Document(content="", meta={"title": "Second", "url": "https://two.example"}),
            Document(content="Third body", meta={}),
        ]

        text = documents_to_text(documents)

        assert text == (
            "1. First\n   https://one.example\n   First body\n\n"
            "2. Second\n   https://two.example\n\n"
            "3. Untitled\n   Third body"
        )
        assert documents_to_text([]) == "No results found."

    @patch("haystack_integrations.components.websearch.youcom.youcom_websearch.request_with_retry")
    def test_as_tool_wraps_component(self, mock_request):
        mock_request.return_value = mock_response(SAMPLE_RESPONSE)

        component = YouComWebSearch(top_k=2)
        tool = component.as_tool()

        assert isinstance(tool, ComponentTool)
        assert tool.name == DEFAULT_TOOL_NAME
        assert tool.description == DEFAULT_TOOL_DESCRIPTION
        assert "query" in tool.parameters["properties"]
        assert tool.parameters["required"] == ["query"]

        result = tool.invoke(query="test")
        assert len(result["documents"]) == 2
        text = tool.outputs_to_string["handler"](result[tool.outputs_to_string["source"]])
        assert text.startswith(
            "1. Haystack | Haystack\n   https://haystack.deepset.ai\n   Haystack is an open-source framework"
        )
        assert "2. AI news item\n   https://example.com/news\n   A news description" in text
        assert mock_request.call_args.kwargs["params"]["count"] == 2

    def test_as_tool_custom_name_and_description(self):
        tool = YouComWebSearch().as_tool(name="web_lookup", description="Look things up.")

        assert tool.name == "web_lookup"
        assert tool.description == "Look things up."

    def test_as_tool_serialization_roundtrip(self):
        tool = YouComWebSearch(top_k=3, extraction_mode="highlights").as_tool()
        data = tool.to_dict()
        restored = ComponentTool.from_dict(data)

        assert isinstance(restored, ComponentTool)
        assert restored.name == DEFAULT_TOOL_NAME
        assert restored._component.top_k == 3
        assert restored._component.extraction_mode == "highlights"
        assert restored.outputs_to_string["handler"] is documents_to_text

    def test_serialization_roundtrip_extraction_params(self):
        component = YouComWebSearch(extraction_mode="full_page", extraction_formats=["html"], crawl_timeout=20)
        data = component_to_dict(component, name="websearch")
        restored = component_from_dict(YouComWebSearch, data, name="websearch")

        assert restored.extraction_mode == "full_page"
        assert restored.extraction_formats == ["html"]
        assert restored.crawl_timeout == 20

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
    def test_keyed_live_search_highlights(self):
        component = YouComWebSearch(top_k=2, extraction_mode="highlights")
        result = component.run(query="What is Haystack by deepset?")

        assert len(result["documents"]) > 0
        assert result["documents"][0].content
        assert result["documents"][0].meta["content_level"] in ("highlights", "snippets")

    @pytest.mark.skipif("YDC_API_KEY" not in os.environ, reason="YDC_API_KEY not set")
    async def test_keyed_live_search_async(self):
        component = YouComWebSearch(top_k=2)
        result = await component.run_async(query="What is Haystack by deepset?")

        assert len(result["documents"]) > 0
