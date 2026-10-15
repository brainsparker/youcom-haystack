# SPDX-FileCopyrightText: 2026-present You.com <developers@you.com>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
from haystack import Document, component, logging
from haystack.utils import Secret
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

YOUCOM_KEYED_SEARCH_URL = "https://ydc-index.io/v1/search"
YOUCOM_KEYLESS_SEARCH_URL = "https://api.you.com/v1/agents/search"

try:
    _VERSION = version("youcom-haystack")
except PackageNotFoundError:  # pragma: no cover
    _VERSION = "0.0.0-dev"

USER_AGENT = f"youcom-haystack/{_VERSION}"

_RATE_LIMIT_STATUSES = (402, 429)
_UPGRADE_HINT = (
    "You.com keyless free-tier limit reached. Set the YDC_API_KEY environment variable "
    "(or pass api_key) for higher limits. Get a free API key at https://you.com/platform."
)

# Transient statuses worth another attempt. 429 is retried on the keyed API only: on the
# keyless tier it means the daily per-IP allowance is spent, and retrying cannot change that.
RETRY_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_RETRY_AFTER_SECONDS = 30.0

# Domain-list parameters of the You.com Search API and the maximum entries each accepts.
DOMAIN_LIST_LIMITS = {"include_domains": 500, "exclude_domains": 5000, "boost_domains": 500}
_DOMAIN_LIST_PARAMS = tuple(DOMAIN_LIST_LIMITS)

# Per-run overrides that have a dedicated keyword argument and their API parameter names.
_RUN_PARAM_NAMES = {"top_k": "count", "freshness": "freshness"}


class YouComSearchError(httpx.HTTPStatusError):
    """
    Raised when the You.com Search API answers a search with an error status.

    Subclasses `httpx.HTTPStatusError`, so existing handlers keep working and `request`
    and `response` stay available for inspection.
    """


class YouComFreeTierLimitError(YouComSearchError):
    """
    Raised when the keyless free tier has reached its per-IP limit.

    Set the `YDC_API_KEY` environment variable (or pass `api_key`) to use the keyed API.
    """


def _normalize_domains(name: str, value: Any) -> list[str]:
    """
    Normalize a domain-list parameter to a list of unique, non-empty domains.

    Accepts a list (or any iterable) of strings, or a comma-separated string.

    :raises ValueError: When the value has the wrong type or exceeds the API limit for `name`.
    """
    if isinstance(value, str):
        raw: Iterable[Any] = value.split(",")
    elif isinstance(value, Iterable):
        raw = value
    else:
        msg = f"{name} must be a list of domain strings or a comma-separated string, got {type(value).__name__}."
        raise ValueError(msg)

    domains: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            msg = f"{name} must contain only strings, got {type(item).__name__}."
            raise ValueError(msg)
        domain = item.strip()
        if domain and domain not in domains:
            domains.append(domain)

    limit = DOMAIN_LIST_LIMITS[name]
    if len(domains) > limit:
        msg = f"{name} supports up to {limit} domains, got {len(domains)}."
        raise ValueError(msg)
    return domains


def _check_domain_lists(params: dict[str, Any]) -> None:
    """
    Enforce the You.com rule that `include_domains` cannot be combined with the other domain lists.

    The API answers such requests with a 422; failing early gives a clearer message and
    spends no request.
    """
    if params.get("include_domains") and (params.get("exclude_domains") or params.get("boost_domains")):
        msg = (
            "include_domains is a strict allowlist and cannot be combined with exclude_domains or boost_domains. "
            "Use include_domains alone, or exclude_domains with boost_domains."
        )
        raise ValueError(msg)


def _retry_wait(retry_state: RetryCallState) -> float:
    """
    Wait as long as the server asked in `Retry-After` (capped), otherwise back off exponentially.
    """
    exponential = wait_exponential(multiplier=1, max=10)(retry_state)
    outcome = retry_state.outcome
    error = outcome.exception() if outcome is not None and outcome.failed else None
    if isinstance(error, httpx.HTTPStatusError):
        retry_after = error.response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), _MAX_RETRY_AFTER_SECONDS)
            except ValueError:
                pass
    return exponential


@component
class YouComWebSearch:
    """
    A component that uses the You.com Search API to search the web and return results as Haystack Documents.

    Works with zero configuration: when no API key is available, searches use You.com's keyless
    free tier (rate limited per IP), so getting-started pipelines run without any setup.
    Set the `YDC_API_KEY` environment variable (or pass `api_key`) to use the keyed
    [You.com Search API](https://you.com/docs/api-reference/search/v1-search) with higher limits.

    The component keeps a pooled HTTP client between calls. `warm_up()` opens it (it is also
    opened on first use) and `close()` releases it; `Pipeline` and `Agent` call both for you.
    The async pair is `warm_up_async()` and `close_async()`.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.youcom import YouComWebSearch

    websearch = YouComWebSearch(top_k=5)  # no API key needed to get started
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]

    # Narrow a single call without rebuilding the component
    recent = websearch.run(query="Haystack release", freshness="week", search_params={"country": "DE"})
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("YDC_API_KEY", strict=False),
        top_k: int | None = 10,
        freshness: str | None = None,
        country: str | None = None,
        search_lang: str | None = None,
        safesearch: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        boost_domains: list[str] | None = None,
        extra_params: dict[str, Any] | None = None,
        timeout: int = 10,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize the YouComWebSearch component.

        :param api_key:
            You.com API key. Defaults to the `YDC_API_KEY` environment variable, resolved
            leniently: when the variable is not set, the component uses You.com's keyless
            free tier (rate limited per IP) instead of raising an error.
        :param top_k:
            Maximum number of results to return per section (web, news). Maps to the
            `count` parameter in the You.com API (1-100).
        :param freshness:
            Only return results from within the given window: `"day"`, `"week"`, `"month"`,
            `"year"`, or a date range in the format `"YYYY-MM-DDtoYYYY-MM-DD"`.
        :param country:
            2-letter country code determining the geographical focus of web results (e.g. `"US"`, `"DE"`).
        :param search_lang:
            Language of the returned web results in BCP 47 format (e.g. `"EN"`, `"PT-BR"`).
            Maps to the `language` parameter in the You.com API.
        :param safesearch:
            Content moderation level: `"off"`, `"moderate"`, or `"strict"`.
        :param include_domains:
            Strict allowlist: only results from these domains are returned (up to 500).
            Cannot be combined with `exclude_domains` or `boost_domains`.
        :param exclude_domains:
            Domains filtered out of the results (up to 5000). Can be combined with `boost_domains`.
        :param boost_domains:
            Domains that receive a fixed relative ranking boost (up to 500). This is not a filter:
            results from other domains can still appear.
        :param extra_params:
            Additional query parameters passed directly to the You.com Search API
            (e.g. `{"offset": 1}`). Domain lists given here are normalized like the dedicated parameters.
        :param timeout:
            Timeout in seconds for the HTTP request. Defaults to 10.
        :param max_retries:
            Maximum number of attempts per search, including the first. Transient failures
            (connection errors, timeouts, HTTP 408, 425, 429, 500, 502, 503, 504) are retried
            with exponential backoff, honoring `Retry-After` when the server sends it. On the
            keyless tier, 402 and 429 mean the daily allowance is spent and are not retried.
            Defaults to 3.
        """
        self.api_key = api_key
        self.top_k = top_k
        self.freshness = freshness
        self.country = country
        self.search_lang = search_lang
        self.safesearch = safesearch
        self.include_domains = include_domains
        self.exclude_domains = exclude_domains
        self.boost_domains = boost_domains
        self.extra_params = extra_params
        self.timeout = timeout
        self.max_retries = max_retries

        # Validate the init-time domain lists now so misconfiguration fails at construction.
        self._build_params(query="", top_k=None, freshness=None, search_params=None)

        self._client: httpx.Client | None = None
        self._async_client: httpx.AsyncClient | None = None

    # Lifecycle

    def warm_up(self) -> None:
        """
        Open the pooled HTTP client used by `run()`.

        Called automatically on first use. `Pipeline.warm_up()` calls it ahead of time to avoid
        cold-start latency on the first search.
        """
        if self._client is None:
            self._client = httpx.Client(headers=self._base_headers(), timeout=self.timeout)

    async def warm_up_async(self) -> None:
        """
        Open the pooled HTTP client used by `run_async()`.

        Called automatically on first use. `Pipeline.warm_up_async()` calls it ahead of time.
        """
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(headers=self._base_headers(), timeout=self.timeout)

    def close(self) -> None:
        """
        Release the pooled HTTP client opened by `warm_up()`.

        Safe to call more than once. A later `run()` reopens the client.
        """
        if self._client is not None:
            self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """
        Release the pooled HTTP client opened by `warm_up_async()`.

        Safe to call more than once. A later `run_async()` reopens the client.
        """
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    # Search

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        top_k: int | None = None,
        freshness: str | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using the You.com Search API and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :param freshness:
            Optional per-run override of the freshness window (`"day"`, `"week"`, `"month"`,
            `"year"`, or `"YYYY-MM-DDtoYYYY-MM-DD"`). If not provided, the init-time `freshness` is used.
        :param search_params:
            Optional per-run overrides of any other You.com Search API parameter, merged over the
            init-time settings for this call only (e.g. `{"country": "DE", "exclude_domains": ["x.com"]}`).
            Set a key to `None` to drop the init-time value for this call.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        :raises YouComFreeTierLimitError: When the keyless free tier has reached its limit.
        :raises YouComSearchError: When the API answers with any other error status.
        """
        self.warm_up()
        assert self._client is not None  # noqa: S101
        params = self._build_params(query=query, top_k=top_k, freshness=freshness, search_params=search_params)
        url, keyed = self._endpoint()
        client = self._client

        try:
            for attempt in self._retrying():
                with attempt:
                    response = client.get(url, params=params, headers=self._auth_headers())
                    self._raise_if_retryable(response, keyed)
        except httpx.HTTPStatusError as error:
            raise self._search_error(error.response, keyed) from error
        if response.is_error:
            raise self._search_error(response, keyed)
        return self._parse_response(response.json())

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
        freshness: str | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using the You.com Search API and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :param freshness:
            Optional per-run override of the freshness window (`"day"`, `"week"`, `"month"`,
            `"year"`, or `"YYYY-MM-DDtoYYYY-MM-DD"`). If not provided, the init-time `freshness` is used.
        :param search_params:
            Optional per-run overrides of any other You.com Search API parameter, merged over the
            init-time settings for this call only (e.g. `{"country": "DE", "exclude_domains": ["x.com"]}`).
            Set a key to `None` to drop the init-time value for this call.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        :raises YouComFreeTierLimitError: When the keyless free tier has reached its limit.
        :raises YouComSearchError: When the API answers with any other error status.
        """
        await self.warm_up_async()
        assert self._async_client is not None  # noqa: S101
        params = self._build_params(query=query, top_k=top_k, freshness=freshness, search_params=search_params)
        url, keyed = self._endpoint()
        client = self._async_client

        try:
            async for attempt in self._async_retrying():
                with attempt:
                    response = await client.get(url, params=params, headers=self._auth_headers())
                    self._raise_if_retryable(response, keyed)
        except httpx.HTTPStatusError as error:
            raise self._search_error(error.response, keyed) from error
        if response.is_error:
            raise self._search_error(response, keyed)
        return self._parse_response(response.json())

    # Request assembly

    @staticmethod
    def _base_headers() -> dict[str, str]:
        return {"Accept": "application/json", "User-Agent": USER_AGENT}

    def _resolved_api_key(self) -> str | None:
        return self.api_key.resolve_value() if self.api_key else None

    def _endpoint(self) -> tuple[str, bool]:
        """
        Select the endpoint based on API key availability.

        With a resolvable API key, requests go to the keyed You.com Search API.
        Without one, requests fall back to the keyless free tier.

        :returns: The URL and whether the request is keyed.
        """
        if self._resolved_api_key():
            return YOUCOM_KEYED_SEARCH_URL, True
        return YOUCOM_KEYLESS_SEARCH_URL, False

    def _auth_headers(self) -> dict[str, str]:
        """
        Per-request headers. The key is resolved on every call so a key set after construction is picked up.
        """
        api_key = self._resolved_api_key()
        return {"X-API-Key": api_key} if api_key else {}

    def _build_params(
        self,
        query: str,
        top_k: int | None,
        freshness: str | None,
        search_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """
        Merge init-time settings, `extra_params`, and per-run overrides into GET query parameters.

        Precedence, lowest to highest: dedicated init parameters, `extra_params`, `search_params`,
        then the dedicated `top_k` and `freshness` run arguments. Domain lists are validated
        against the You.com limits and sent comma-separated.
        """
        params: dict[str, Any] = {}
        if self.top_k is not None:
            params["count"] = self.top_k
        if self.freshness is not None:
            params["freshness"] = self.freshness
        if self.country is not None:
            params["country"] = self.country
        if self.search_lang is not None:
            params["language"] = self.search_lang
        if self.safesearch is not None:
            params["safesearch"] = self.safesearch
        for name in _DOMAIN_LIST_PARAMS:
            value = getattr(self, name)
            if value is not None:
                params[name] = value
        if self.extra_params:
            params.update(self.extra_params)
        if search_params:
            if "query" in search_params:
                msg = "search_params cannot override the query; pass it as the query argument."
                raise ValueError(msg)
            params.update(search_params)
        if top_k is not None:
            params["count"] = top_k
        if freshness is not None:
            params["freshness"] = freshness

        # Drop keys cleared with None, normalize domain lists, and enforce the API's combination rule.
        params = {key: value for key, value in params.items() if value is not None}
        for name in _DOMAIN_LIST_PARAMS:
            if name in params:
                params[name] = _normalize_domains(name, params[name])
                if not params[name]:
                    del params[name]
        _check_domain_lists(params)

        for name in _DOMAIN_LIST_PARAMS:
            if name in params:
                params[name] = ",".join(params[name])
        return {"query": query, **params}

    # Retries and errors

    def _retrying(self) -> Retrying:
        return Retrying(**self._retry_kwargs())

    def _async_retrying(self) -> AsyncRetrying:
        return AsyncRetrying(**self._retry_kwargs())

    def _retry_kwargs(self) -> dict[str, Any]:
        return {
            "reraise": True,
            "stop": stop_after_attempt(max(self.max_retries, 1)),
            "wait": _retry_wait,
            "retry": retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError)),
        }

    @staticmethod
    def _is_free_tier_limit(status: int, keyed: bool) -> bool:
        return status in _RATE_LIMIT_STATUSES and not keyed

    def _raise_if_retryable(self, response: httpx.Response, keyed: bool) -> None:
        """
        Raise for statuses that deserve another attempt so the retry loop picks them up.

        Other error statuses are returned to the caller and reported once, without retries.
        """
        status = response.status_code
        if status in RETRY_STATUSES and not self._is_free_tier_limit(status, keyed):
            logger.debug("You.com search returned HTTP {status}, retrying", status=status)
            response.raise_for_status()

    def _search_error(self, response: httpx.Response, keyed: bool) -> YouComSearchError:
        """
        Build the error for a failed search, with upgrade guidance when the keyless tier is exhausted.
        """
        status = response.status_code
        detail = response.text.strip()
        message = f"You.com search failed with HTTP {status} for {response.request.url}"
        if detail:
            message = f"{message}: {detail[:500]}"
        if self._is_free_tier_limit(status, keyed):
            return YouComFreeTierLimitError(f"{message}. {_UPGRADE_HINT}", request=response.request, response=response)
        return YouComSearchError(message, request=response.request, response=response)

    # Response parsing

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a You.com Search API response to Haystack Documents and links.

        Web results come first, then news results. Document content is the joined
        text snippets when available, falling back to the result description.

        :param response: You.com Search API response dictionary.
        :returns: Dictionary with `documents` and `links` keys.
        """
        documents: list[Document] = []
        links: list[str] = []

        results = response.get("results", {})
        for section in ("web", "news"):
            for result in results.get(section, []):
                url = result.get("url", "")
                title = result.get("title", "")
                snippets = result.get("snippets") or []
                content = "\n".join(snippets) if snippets else result.get("description", "")

                meta: dict[str, Any] = {"title": title, "url": url, "source": section}
                if result.get("page_age"):
                    meta["page_age"] = result["page_age"]

                documents.append(Document(content=content, meta=meta))
                if url:
                    links.append(url)

        return {"documents": documents, "links": links}
