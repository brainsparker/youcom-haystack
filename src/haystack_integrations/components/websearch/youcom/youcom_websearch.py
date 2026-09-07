# SPDX-FileCopyrightText: 2026-present You.com <developers@you.com>
#
# SPDX-License-Identifier: Apache-2.0

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from haystack import Document, component, logging
from haystack.tools import ComponentTool
from haystack.utils import Secret
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry
from requests.exceptions import HTTPError

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
    "(or pass api_key) for higher limits — get a free API key at https://you.com/platform."
)

EXTRACTION_MODES = ("snippets", "highlights", "full_page")
EXTRACTION_FORMATS = ("markdown", "html")
_EXTRACTION_KEYLESS_HINT = (
    "extraction_mode='{mode}' requires the keyed You.com Search API: the keyless free tier returns snippets only. "
    "Set the YDC_API_KEY environment variable (or pass api_key), get a free API key at https://you.com/platform, "
    "or use extraction_mode='snippets'."
)
# Domain-list parameters are comma-separated strings on GET and JSON arrays on POST.
_DOMAIN_LIST_PARAMS = ("include_domains", "exclude_domains", "boost_domains")

DEFAULT_TOOL_NAME = "youcom_web_search"
DEFAULT_TOOL_DESCRIPTION = (
    "Search the web with You.com and return the most relevant pages and news articles, "
    "each with its title, URL, and content. Use it to look up current information, "
    "verify facts, or find documentation and sources."
)


def documents_to_text(documents: list[Document]) -> str:
    """
    Format search result Documents as a numbered list for a language model.

    Used as the default `outputs_to_string` handler of the tool returned by
    `YouComWebSearch.as_tool()`. Each entry carries the title, URL, and content
    of one result so the model can ground and cite its answer.

    :param documents: Documents returned by `YouComWebSearch.run()`.
    :returns: The formatted results, or a short notice when there are none.
    """
    if not documents:
        return "No results found."
    lines: list[str] = []
    for index, doc in enumerate(documents, start=1):
        title = doc.meta.get("title") or "Untitled"
        url = doc.meta.get("url") or ""
        header = f"{index}. {title}"
        if url:
            header = f"{header}\n   {url}"
        content = (doc.content or "").strip()
        lines.append(f"{header}\n   {content}" if content else header)
    return "\n\n".join(lines)


@component
class YouComWebSearch:
    """
    A component that uses the You.com Search API to search the web and return results as Haystack Documents.

    Works with zero configuration: when no API key is available, searches use You.com's keyless
    free tier (rate limited per IP), so getting-started pipelines run without any setup.
    Set the `YDC_API_KEY` environment variable (or pass `api_key`) to use the keyed
    [You.com Search API](https://you.com/docs/api-reference/search/v1-search) with higher limits.

    Each result can carry one of three content levels, selected with `extraction_mode`:
    `"snippets"` (default, short keyword-centered fragments), `"highlights"` (the passages of each
    page that address the query, sized for agents), or `"full_page"` (the whole page as Markdown or
    HTML). Highlights and full-page content require an API key.

    ### Usage example

    ```python
    from haystack_integrations.components.websearch.youcom import YouComWebSearch

    websearch = YouComWebSearch(top_k=5)  # no API key needed to get started
    result = websearch.run(query="What is Haystack by deepset?")
    documents = result["documents"]
    links = result["links"]
    ```

    ### Use as an Agent tool

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack_integrations.components.websearch.youcom import YouComWebSearch

    search_tool = YouComWebSearch(top_k=5, extraction_mode="highlights").as_tool()
    agent = Agent(chat_generator=OpenAIChatGenerator(), tools=[search_tool])
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
        extra_params: dict[str, Any] | None = None,
        timeout: int = 10,
        max_retries: int = 3,
        extraction_mode: str = "snippets",
        extraction_formats: list[str] | None = None,
        crawl_timeout: int | None = None,
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
        :param extra_params:
            Additional query parameters passed directly to the You.com Search API
            (e.g. `{"include_domains": "nytimes.com,bbc.com"}`). Comma-separated domain lists
            are converted to JSON arrays when the request is sent as `POST`.
        :param timeout:
            Timeout in seconds for the HTTP request. Defaults to 10.
        :param max_retries:
            Maximum number of retry attempts on transient failures. Defaults to 3.
        :param extraction_mode:
            Content level of each result: `"snippets"` (default) returns short keyword-centered
            fragments; `"highlights"` returns the passages of each page most relevant to the query;
            `"full_page"` returns the whole page. Highlights and full-page content are requested
            through `POST /v1/search` and require an API key; the keyless free tier returns
            snippets only. Full-page extraction is billed separately by You.com.
        :param extraction_formats:
            Formats of the full-page content, any of `"markdown"` and `"html"`. Only used with
            `extraction_mode="full_page"`. Defaults to the API default (`["markdown"]`).
            Document content uses Markdown when present, otherwise HTML.
        :param crawl_timeout:
            Maximum time in seconds (1-60) to wait for page content when the request crawls pages
            (`extraction_mode="full_page"`). Defaults to the API default (10 seconds).
        """
        if extraction_mode not in EXTRACTION_MODES:
            msg = f"extraction_mode must be one of {EXTRACTION_MODES}, got {extraction_mode!r}."
            raise ValueError(msg)
        if extraction_formats is not None:
            invalid = [f for f in extraction_formats if f not in EXTRACTION_FORMATS]
            if not extraction_formats or invalid:
                msg = (
                    f"extraction_formats must be a non-empty list drawn from {EXTRACTION_FORMATS}, "
                    f"got {extraction_formats!r}."
                )
                raise ValueError(msg)

        self.api_key = api_key
        self.top_k = top_k
        self.freshness = freshness
        self.country = country
        self.search_lang = search_lang
        self.safesearch = safesearch
        self.extra_params = extra_params
        self.timeout = timeout
        self.max_retries = max_retries
        self.extraction_mode = extraction_mode
        self.extraction_formats = extraction_formats
        self.crawl_timeout = crawl_timeout

    @component.output_types(documents=list[Document], links=list[str])
    def run(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Search the web using the You.com Search API and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        request = self._build_request_kwargs(query=query, top_k=top_k)

        try:
            response = request_with_retry(attempts=self.max_retries, timeout=self.timeout, **request)
            response.raise_for_status()
        except HTTPError as error:
            raise self._enrich_error(error, request["headers"]) from error
        return self._parse_response(response.json())

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(
        self,
        query: str,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """
        Asynchronously search the web using the You.com Search API and return results as Documents.

        :param query: Search query string.
        :param top_k:
            Optional per-run override of the maximum number of results.
            If not provided, the init-time `top_k` is used.
        :returns: A dictionary with:
            - `documents`: List of Documents containing search result content.
            - `links`: List of URLs from the search results.
        """
        request = self._build_request_kwargs(query=query, top_k=top_k)

        try:
            response = await async_request_with_retry(attempts=self.max_retries, timeout=self.timeout, **request)
            response.raise_for_status()
        except HTTPError as error:
            raise self._enrich_error(error, request["headers"]) from error
        return self._parse_response(response.json())

    def as_tool(
        self,
        name: str = DEFAULT_TOOL_NAME,
        description: str = DEFAULT_TOOL_DESCRIPTION,
    ) -> ComponentTool:
        """
        Wrap this component as a Haystack `Tool` for use with `Agent`.

        The tool takes the `query` (and optional `top_k`) parameters of `run()` and returns the
        results to the model as a numbered list of title, URL, and content (see `documents_to_text`).
        Choose `extraction_mode="highlights"` on the component for query-relevant passages instead of
        snippets when the results are read by a model rather than a person.

        :param name: Tool name shown to the model. Defaults to `"youcom_web_search"`.
        :param description: Tool description shown to the model.
        :returns: A `ComponentTool` wrapping this component.
        """
        return ComponentTool(
            component=self,
            name=name,
            description=description,
            outputs_to_string={"source": "documents", "handler": documents_to_text},
        )

    def _build_request(self) -> tuple[str, dict[str, str]]:
        """
        Select the endpoint and headers based on API key availability.

        With a resolvable API key, requests go to the keyed You.com Search API.
        Without one, requests fall back to the keyless free tier.
        """
        headers = {
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        api_key = self.api_key.resolve_value() if self.api_key else None
        if api_key:
            headers["X-API-Key"] = api_key
            return YOUCOM_KEYED_SEARCH_URL, headers
        return YOUCOM_KEYLESS_SEARCH_URL, headers

    def _build_request_kwargs(self, query: str, top_k: int | None) -> dict[str, Any]:
        """
        Assemble the keyword arguments of the HTTP request.

        Snippet searches use `GET` with query parameters on both tiers. Highlights and full-page
        content are only available on `POST /v1/search` of the keyed API, where all parameters are
        sent as a JSON body.
        """
        url, headers = self._build_request()
        params = self._build_params(query=query, top_k=top_k)

        if self.extraction_mode == "snippets":
            return {"method": "GET", "url": url, "params": params, "headers": headers}

        if "X-API-Key" not in headers:
            raise ValueError(_EXTRACTION_KEYLESS_HINT.format(mode=self.extraction_mode))

        body = self._to_post_body(params)
        extraction: dict[str, Any] = {"extraction_mode": self.extraction_mode}
        if self.extraction_mode == "full_page" and self.extraction_formats is not None:
            extraction["full_page"] = {"extraction_formats": list(self.extraction_formats)}
        body["extraction"] = extraction
        if self.crawl_timeout is not None:
            body["crawl_timeout"] = self.crawl_timeout
        return {"method": "POST", "url": url, "json": body, "headers": headers}

    def _build_params(self, query: str, top_k: int | None) -> dict[str, Any]:
        effective_top_k = top_k if top_k is not None else self.top_k
        params: dict[str, Any] = {"query": query}
        if effective_top_k is not None:
            params["count"] = effective_top_k
        if self.freshness is not None:
            params["freshness"] = self.freshness
        if self.country is not None:
            params["country"] = self.country
        if self.search_lang is not None:
            params["language"] = self.search_lang
        if self.safesearch is not None:
            params["safesearch"] = self.safesearch
        if self.extra_params:
            params.update(self.extra_params)
        return params

    @staticmethod
    def _to_post_body(params: dict[str, Any]) -> dict[str, Any]:
        """
        Convert GET-style query parameters to a `POST /v1/search` JSON body.

        Domain lists are comma-separated strings on `GET` and JSON arrays on `POST`.
        """
        body = dict(params)
        for key in _DOMAIN_LIST_PARAMS:
            value = body.get(key)
            if isinstance(value, str):
                body[key] = [domain.strip() for domain in value.split(",") if domain.strip()]
        return body

    @staticmethod
    def _enrich_error(error: HTTPError, headers: dict[str, str]) -> HTTPError:
        """
        Add upgrade guidance to keyless rate-limit errors; return other errors unchanged.
        """
        status = error.response.status_code if error.response is not None else None
        if status in _RATE_LIMIT_STATUSES and "X-API-Key" not in headers:
            return HTTPError(f"{error} — {_UPGRADE_HINT}", response=error.response)
        return error

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a You.com Search API response to Haystack Documents and links.

        Web results come first, then news results. Document content depends on `extraction_mode`:
        the joined highlights, the full page (Markdown, else HTML), or the joined snippets, in each
        case falling back to the next available level and finally to the result description.

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
                content, level, content_format = self._extract_content(result)

                meta: dict[str, Any] = {"title": title, "url": url, "source": section}
                if result.get("page_age"):
                    meta["page_age"] = result["page_age"]
                if self.extraction_mode != "snippets":
                    meta["content_level"] = level
                    if content_format is not None:
                        meta["content_format"] = content_format

                documents.append(Document(content=content, meta=meta))
                if url:
                    links.append(url)

        return {"documents": documents, "links": links}

    def _extract_content(self, result: dict[str, Any]) -> tuple[str, str, str | None]:
        """
        Pick the Document content of one result according to `extraction_mode`.

        :returns: The content, the content level actually used (`"snippets"`, `"highlights"`,
            or `"full_page"`), and the format of full-page content (`"markdown"` or `"html"`).
        """
        contents = result.get("contents") or {}
        if self.extraction_mode == "full_page":
            for content_format in EXTRACTION_FORMATS:
                page = contents.get(content_format)
                if isinstance(page, str) and page:
                    return page, "full_page", content_format
        if self.extraction_mode in ("highlights", "full_page"):
            highlights = contents.get("highlights") or []
            if highlights:
                return "\n".join(highlights), "highlights", None
        snippets = result.get("snippets") or []
        content = "\n".join(snippets) if snippets else result.get("description", "")
        return content, "snippets", None
