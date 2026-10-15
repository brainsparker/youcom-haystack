# SPDX-FileCopyrightText: 2026-present You.com <developers@you.com>
#
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable

import httpx
import pytest
from haystack import Document, Pipeline
from haystack.core.serialization import component_from_dict, component_to_dict
from haystack.utils import Secret
from tenacity import RetryCallState

from haystack_integrations.components.websearch.youcom import (
    YouComFreeTierLimitError,
    YouComSearchError,
    YouComWebSearch,
)
from haystack_integrations.components.websearch.youcom import youcom_websearch as module
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


class FakeAPI:
    """
    Records every request the component sends and answers from a queue of responses.

    The last queued response is repeated once the queue is exhausted.
    """

    def __init__(self, *responses: httpx.Response | Exception) -> None:
        self.responses = list(responses) or [httpx.Response(200, json=SAMPLE_RESPONSE)]
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


@pytest.fixture
def api(monkeypatch) -> Callable[..., FakeAPI]:
    """Route the component's pooled clients through an in-memory You.com API."""

    def install(*responses: httpx.Response | Exception) -> FakeAPI:
        fake = FakeAPI(*responses)

        class Client(httpx.Client):
            def __init__(self, **kwargs) -> None:
                super().__init__(transport=fake.transport, **kwargs)

        class AsyncClient(httpx.AsyncClient):
            def __init__(self, **kwargs) -> None:
                super().__init__(transport=fake.transport, **kwargs)

        monkeypatch.setattr(module.httpx, "Client", Client)
        monkeypatch.setattr(module.httpx, "AsyncClient", AsyncClient)
        return fake

    return install


@pytest.fixture(autouse=True)
def no_retry_wait(monkeypatch):
    monkeypatch.setattr(module, "_retry_wait", lambda _retry_state: 0)


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
        assert component.include_domains is None
        assert component.exclude_domains is None
        assert component.boost_domains is None
        assert component.extra_params is None
        assert component.timeout == 10
        assert component.max_retries == 3

    def test_init_without_env_var_does_not_raise(self):
        component = YouComWebSearch()
        assert component.api_key.resolve_value() is None

    def test_run_keyless_when_no_api_key(self, api):
        fake = api()
        YouComWebSearch().run(query="test")

        assert str(fake.last.url).startswith(YOUCOM_KEYLESS_SEARCH_URL)
        assert "X-API-Key" not in fake.last.headers

    def test_run_keyed_when_env_var_set(self, api, monkeypatch):
        monkeypatch.setenv("YDC_API_KEY", "env-key")
        fake = api()
        YouComWebSearch().run(query="test")

        assert str(fake.last.url).startswith(YOUCOM_KEYED_SEARCH_URL)
        assert fake.last.headers["X-API-Key"] == "env-key"

    def test_run_keyed_with_explicit_token(self, api):
        fake = api()
        YouComWebSearch(api_key=Secret.from_token("explicit-key")).run(query="test")

        assert str(fake.last.url).startswith(YOUCOM_KEYED_SEARCH_URL)
        assert fake.last.headers["X-API-Key"] == "explicit-key"

    def test_run_picks_up_key_set_after_warm_up(self, api, monkeypatch):
        fake = api()
        component = YouComWebSearch()
        component.warm_up()
        monkeypatch.setenv("YDC_API_KEY", "late-key")
        component.run(query="test")

        assert str(fake.last.url).startswith(YOUCOM_KEYED_SEARCH_URL)
        assert fake.last.headers["X-API-Key"] == "late-key"

    def test_run_sends_user_agent(self, api):
        fake = api()
        YouComWebSearch().run(query="test")

        assert fake.last.headers["User-Agent"] == USER_AGENT
        assert USER_AGENT.startswith("youcom-haystack/")

    def test_run_builds_params(self, api):
        fake = api()
        component = YouComWebSearch(
            top_k=5,
            freshness="week",
            country="US",
            search_lang="EN",
            safesearch="moderate",
            extra_params={"offset": 1},
        )
        component.run(query="climate news")

        assert dict(fake.last.url.params) == {
            "query": "climate news",
            "count": "5",
            "freshness": "week",
            "country": "US",
            "language": "EN",
            "safesearch": "moderate",
            "offset": "1",
        }

    def test_run_top_k_override(self, api):
        fake = api()
        YouComWebSearch(top_k=10).run(query="q", top_k=3)

        assert fake.last.url.params["count"] == "3"

    def test_run_freshness_override(self, api):
        fake = api()
        component = YouComWebSearch(freshness="year")
        component.run(query="q", freshness="day")
        assert fake.last.url.params["freshness"] == "day"

        component.run(query="q")
        assert fake.last.url.params["freshness"] == "year"

    def test_run_search_params_override_is_per_call(self, api):
        fake = api()
        component = YouComWebSearch(country="US", safesearch="strict")

        component.run(query="q", search_params={"country": "DE", "safesearch": None, "offset": 2})
        params = dict(fake.last.url.params)
        assert params["country"] == "DE"
        assert "safesearch" not in params
        assert params["offset"] == "2"

        component.run(query="q")
        params = dict(fake.last.url.params)
        assert params["country"] == "US"
        assert params["safesearch"] == "strict"
        assert "offset" not in params

    def test_run_dedicated_arguments_win_over_search_params(self, api):
        fake = api()
        YouComWebSearch().run(query="q", top_k=2, freshness="day", search_params={"count": 50, "freshness": "year"})

        params = dict(fake.last.url.params)
        assert params["count"] == "2"
        assert params["freshness"] == "day"

    def test_run_search_params_cannot_replace_query(self, api):
        api()
        with pytest.raises(ValueError, match="query"):
            YouComWebSearch().run(query="q", search_params={"query": "other"})

    def test_run_parses_documents_and_links(self, api):
        api()
        result = YouComWebSearch().run(query="test")

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

    def test_run_empty_results(self, api):
        api(httpx.Response(200, json={"results": {"web": [], "news": []}}))
        assert YouComWebSearch().run(query="test") == {"documents": [], "links": []}

    def test_run_missing_sections(self, api):
        api(httpx.Response(200, json={"metadata": {}}))
        assert YouComWebSearch().run(query="test") == {"documents": [], "links": []}

    async def test_run_async(self, api):
        fake = api()
        component = YouComWebSearch()
        result = await component.run_async(query="test", freshness="week")

        assert len(result["documents"]) == 2
        assert str(fake.last.url).startswith(YOUCOM_KEYLESS_SEARCH_URL)
        assert fake.last.url.params["freshness"] == "week"
        await component.close_async()


class TestDomainFilters:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)

    def test_domain_lists_are_sent_comma_separated(self, api):
        fake = api()
        component = YouComWebSearch(
            exclude_domains=["pinterest.com", " quora.com ", "pinterest.com"], boost_domains=["docs.python.org"]
        )
        component.run(query="q")

        params = dict(fake.last.url.params)
        assert params["exclude_domains"] == "pinterest.com,quora.com"
        assert params["boost_domains"] == "docs.python.org"

    def test_include_domains_alone(self, api):
        fake = api()
        YouComWebSearch(include_domains=["nytimes.com", "bbc.com"]).run(query="q")

        assert fake.last.url.params["include_domains"] == "nytimes.com,bbc.com"

    def test_extra_params_domain_string_is_normalized(self, api):
        fake = api()
        YouComWebSearch(extra_params={"include_domains": "nytimes.com, bbc.com,"}).run(query="q")

        assert fake.last.url.params["include_domains"] == "nytimes.com,bbc.com"

    @pytest.mark.parametrize("other", ["exclude_domains", "boost_domains"])
    def test_include_domains_cannot_combine_at_init(self, other):
        with pytest.raises(ValueError, match="include_domains"):
            YouComWebSearch(include_domains=["a.com"], **{other: ["b.com"]})

    def test_include_domains_cannot_combine_per_run(self, api):
        fake = api()
        component = YouComWebSearch(exclude_domains=["b.com"])
        with pytest.raises(ValueError, match="include_domains"):
            component.run(query="q", search_params={"include_domains": ["a.com"]})
        assert fake.requests == []

    def test_per_run_can_clear_init_list(self, api):
        fake = api()
        component = YouComWebSearch(exclude_domains=["b.com"])
        component.run(query="q", search_params={"exclude_domains": None, "include_domains": ["a.com"]})

        params = dict(fake.last.url.params)
        assert params["include_domains"] == "a.com"
        assert "exclude_domains" not in params

    def test_empty_domain_list_is_dropped(self, api):
        fake = api()
        YouComWebSearch(include_domains=[]).run(query="q")
        assert "include_domains" not in fake.last.url.params

    def test_domain_list_limits(self):
        with pytest.raises(ValueError, match="up to 500"):
            YouComWebSearch(include_domains=[f"d{i}.com" for i in range(501)])
        with pytest.raises(ValueError, match="up to 5000"):
            YouComWebSearch(exclude_domains=[f"d{i}.com" for i in range(5001)])
        YouComWebSearch(exclude_domains=[f"d{i}.com" for i in range(5000)])

    def test_domain_list_type_errors(self):
        with pytest.raises(ValueError, match="list of domain strings"):
            YouComWebSearch(include_domains=42)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="only strings"):
            YouComWebSearch(boost_domains=["a.com", 1])  # type: ignore[list-item]


class TestErrorsAndRetries:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)

    @pytest.mark.parametrize("status", [402, 429])
    def test_keyless_limit_raises_upgrade_hint_without_retrying(self, api, status):
        fake = api(httpx.Response(status, text="limit reached"))
        with pytest.raises(YouComFreeTierLimitError, match="YDC_API_KEY") as excinfo:
            YouComWebSearch().run(query="test")

        assert len(fake.requests) == 1
        assert excinfo.value.response.status_code == status
        assert isinstance(excinfo.value, httpx.HTTPStatusError)

    async def test_keyless_limit_raises_upgrade_hint_async(self, api):
        fake = api(httpx.Response(429))
        with pytest.raises(YouComFreeTierLimitError, match="YDC_API_KEY"):
            await YouComWebSearch().run_async(query="test")
        assert len(fake.requests) == 1

    def test_keyed_error_is_not_rewritten(self, api):
        api(httpx.Response(401, json={"error": "bad key"}))
        with pytest.raises(YouComSearchError, match="401") as excinfo:
            YouComWebSearch(api_key=Secret.from_token("bad-key")).run(query="test")

        assert "YDC_API_KEY" not in str(excinfo.value)
        assert not isinstance(excinfo.value, YouComFreeTierLimitError)
        assert "bad key" in str(excinfo.value)

    def test_keyed_429_is_retried(self, api):
        fake = api(httpx.Response(429), httpx.Response(200, json=SAMPLE_RESPONSE))
        result = YouComWebSearch(api_key=Secret.from_token("key"), max_retries=3).run(query="test")

        assert len(result["documents"]) == 2
        assert len(fake.requests) == 2

    def test_server_errors_are_retried_up_to_max_retries(self, api):
        fake = api(httpx.Response(503), httpx.Response(502), httpx.Response(500, text="boom"))
        with pytest.raises(YouComSearchError, match="500") as excinfo:
            YouComWebSearch(max_retries=3).run(query="test")

        assert len(fake.requests) == 3
        assert excinfo.value.response.status_code == 500

    def test_connection_errors_are_retried(self, api):
        fake = api(httpx.ConnectError("refused"), httpx.Response(200, json=SAMPLE_RESPONSE))
        result = YouComWebSearch(max_retries=2).run(query="test")

        assert len(result["documents"]) == 2
        assert len(fake.requests) == 2

    def test_connection_error_is_raised_after_last_attempt(self, api):
        fake = api(httpx.ConnectError("refused"))
        with pytest.raises(httpx.ConnectError):
            YouComWebSearch(max_retries=2).run(query="test")
        assert len(fake.requests) == 2

    def test_client_errors_are_not_retried(self, api):
        fake = api(httpx.Response(422, json={"detail": "include_domains and exclude_domains"}))
        with pytest.raises(YouComSearchError, match="422"):
            YouComWebSearch(max_retries=3).run(query="test")
        assert len(fake.requests) == 1

    async def test_server_errors_are_retried_async(self, api):
        fake = api(httpx.Response(503), httpx.Response(200, json=SAMPLE_RESPONSE))
        result = await YouComWebSearch(max_retries=2).run_async(query="test")

        assert len(result["documents"]) == 2
        assert len(fake.requests) == 2

    def test_retry_after_header_is_honored_and_capped(self, monkeypatch):
        monkeypatch.undo()  # restore the real wait function

        def wait_for(headers: dict[str, str]) -> float:
            request = httpx.Request("GET", YOUCOM_KEYED_SEARCH_URL)
            response = httpx.Response(429, headers=headers, request=request)
            error = httpx.HTTPStatusError("429", request=request, response=response)
            state = RetryCallState(None, None, (), {})
            state.attempt_number = 1
            state.set_exception((type(error), error, None))
            return module._retry_wait(state)

        assert wait_for({"Retry-After": "2"}) == 2.0
        assert wait_for({"Retry-After": "600"}) == module._MAX_RETRY_AFTER_SECONDS
        assert wait_for({"Retry-After": "soon"}) == wait_for({})


class TestLifecycle:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)

    def test_warm_up_is_idempotent_and_close_releases(self, api):
        api()
        component = YouComWebSearch()
        assert component._client is None

        component.warm_up()
        client = component._client
        assert client is not None
        component.warm_up()
        assert component._client is client

        component.close()
        assert component._client is None
        assert client.is_closed
        component.close()  # no-op

    def test_run_reuses_one_client_and_reopens_after_close(self, api):
        api()
        component = YouComWebSearch()
        component.run(query="a")
        client = component._client
        component.run(query="b")
        assert component._client is client

        component.close()
        component.run(query="c")
        assert component._client is not None
        assert component._client is not client
        component.close()

    async def test_async_lifecycle(self, api):
        api()
        component = YouComWebSearch()
        await component.warm_up_async()
        client = component._async_client
        assert client is not None
        await component.run_async(query="a")
        assert component._async_client is client

        await component.close_async()
        assert component._async_client is None
        assert client.is_closed
        await component.close_async()  # no-op

    def test_pipeline_warms_up_and_closes_component(self, api):
        if not hasattr(Pipeline, "close"):
            pytest.skip("Pipeline.close() arrived in Haystack 3.0")
        api()
        component = YouComWebSearch()
        pipeline = Pipeline()
        pipeline.add_component("search", component)

        result = pipeline.run({"search": {"query": "test", "freshness": "week"}})
        assert len(result["search"]["documents"]) == 2
        assert component._client is not None

        pipeline.close()
        assert component._client is None

    def test_client_carries_timeout_and_base_headers(self, api):
        api()
        component = YouComWebSearch(timeout=3)
        component.warm_up()
        assert component._client is not None
        assert component._client.timeout == httpx.Timeout(3)
        assert component._client.headers["User-Agent"] == USER_AGENT
        assert component._client.headers["Accept"] == "application/json"
        component.close()


class TestSerialization:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)

    def test_serialization_roundtrip_without_env_var(self):
        component = YouComWebSearch(top_k=7, freshness="month", exclude_domains=["a.com"], boost_domains=["b.com"])
        data = component_to_dict(component, name="websearch")
        assert "_client" not in data["init_parameters"]
        restored = component_from_dict(YouComWebSearch, data, name="websearch")

        assert restored.top_k == 7
        assert restored.freshness == "month"
        assert restored.exclude_domains == ["a.com"]
        assert restored.boost_domains == ["b.com"]
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
        try:
            result = component.run(query="What is Haystack by deepset?")
        except YouComFreeTierLimitError:
            pytest.skip("keyless free-tier limit reached for this IP")
        finally:
            component.close()

        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0
        assert result["documents"][0].content

    @pytest.mark.skipif("YDC_API_KEY" not in os.environ, reason="YDC_API_KEY not set")
    def test_keyed_live_search(self):
        component = YouComWebSearch(top_k=2)
        result = component.run(query="What is Haystack by deepset?")
        component.close()

        assert len(result["documents"]) > 0
        assert len(result["links"]) > 0

    @pytest.mark.skipif("YDC_API_KEY" not in os.environ, reason="YDC_API_KEY not set")
    def test_keyed_live_search_with_domain_filter(self):
        component = YouComWebSearch(top_k=3, include_domains=["haystack.deepset.ai"])
        result = component.run(query="Haystack pipelines")
        component.close()

        assert len(result["documents"]) > 0
        assert all("haystack.deepset.ai" in doc.meta["url"] for doc in result["documents"])

    @pytest.mark.skipif("YDC_API_KEY" not in os.environ, reason="YDC_API_KEY not set")
    async def test_keyed_live_search_async(self):
        component = YouComWebSearch(top_k=2)
        result = await component.run_async(query="What is Haystack by deepset?")
        await component.close_async()

        assert len(result["documents"]) > 0
