# youcom-haystack

[![PyPI - Version](https://img.shields.io/pypi/v/youcom-haystack.svg)](https://pypi.org/project/youcom-haystack)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/youcom-haystack.svg)](https://pypi.org/project/youcom-haystack)

[You.com](https://you.com) web search for [Haystack](https://haystack.deepset.ai) pipelines and agents.

Works with **zero configuration**: without an API key, searches use You.com's keyless free tier (rate limited per IP), so `pip install` + run yields real search results immediately. Set the `YDC_API_KEY` environment variable to use the keyed [You.com Search API](https://you.com/docs/api-reference/search/v1-search) with higher limits — get a free key at [you.com/platform](https://you.com/platform).

## Installation

```bash
pip install youcom-haystack
```

## Usage

### Standalone

```python
from haystack_integrations.components.websearch.youcom import YouComWebSearch

websearch = YouComWebSearch(top_k=5)  # no API key needed to get started
result = websearch.run(query="What is Haystack by deepset?")

for doc in result["documents"]:
    print(doc.meta["title"], doc.meta["url"])
    print(doc.content)
```

### In a RAG pipeline

```python
from haystack import Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.websearch.youcom import YouComWebSearch

template = [
    ChatMessage.from_user(
        "Answer the question using the retrieved web results.\n"
        "Results:\n{% for doc in documents %}{{ doc.content }}\n{% endfor %}\n"
        "Question: {{ question }}"
    )
]

pipeline = Pipeline()
pipeline.add_component("search", YouComWebSearch(top_k=5))
pipeline.add_component("prompt_builder", ChatPromptBuilder(template=template))
pipeline.add_component("llm", OpenAIChatGenerator())
pipeline.connect("search.documents", "prompt_builder.documents")
pipeline.connect("prompt_builder.prompt", "llm.messages")

question = "What are the latest developments in open-source AI frameworks?"
result = pipeline.run({"search": {"query": question}, "prompt_builder": {"question": question}})
print(result["llm"]["replies"][0].text)
```

### As an Agent tool

`as_tool()` wraps the component as a Haystack `Tool` for [`Agent`](https://docs.haystack.deepset.ai/docs/agent). The model receives the results as a numbered list of title, URL, and content. Pair it with `extraction_mode="highlights"` so each result carries the passages that answer the query instead of keyword snippets (requires an API key, see below).

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.websearch.youcom import YouComWebSearch

search_tool = YouComWebSearch(top_k=5, extraction_mode="highlights").as_tool()
agent = Agent(
    chat_generator=OpenAIChatGenerator(),
    tools=[search_tool],
    system_prompt="Answer with the web search tool and cite the URLs you used.",
)
result = agent.run(messages=[ChatMessage.from_user("What changed in Haystack 3.0?")])
print(result["last_message"].text)
```

`as_tool(name=..., description=...)` overrides what the model sees. The tool serializes with the rest of the Agent (`tool.to_dict()` / `ComponentTool.from_dict()`).

### Content levels: snippets, highlights, full page

Each result can carry one of three levels of text, chosen with `extraction_mode`:

| `extraction_mode` | Document content | Best for | Tier |
| --- | --- | --- | --- |
| `"snippets"` (default) | Short keyword-centered fragments | A person skimming results | keyless or keyed |
| `"highlights"` | The passages of each page that address the query | Agents that read every result | keyed |
| `"full_page"` | The whole page as Markdown (or HTML) | Archiving, full-text analysis, long-context models | keyed, billed separately |

```python
websearch = YouComWebSearch(extraction_mode="highlights")
result = websearch.run(query="Haystack 3 Agent hooks")
for doc in result["documents"]:
    print(doc.meta["content_level"], doc.meta["url"])  # highlights, or snippets when a page had none
    print(doc.content)

archive = YouComWebSearch(extraction_mode="full_page", extraction_formats=["markdown"], crawl_timeout=30)
```

Highlights and full-page content use `POST /v1/search` of the keyed You.com Search API, so they need `YDC_API_KEY`; with no key the component raises a `ValueError` explaining how to upgrade. When a page has no extracted content, the Document falls back to snippets (and then to the description) and `meta["content_level"]` says which level was used. Full-page Documents also carry `meta["content_format"]` (`markdown` or `html`).

### Async

```python
result = await websearch.run_async(query="What is Haystack by deepset?")
```

## Parameters

| Init parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `api_key` | `Secret` | `Secret.from_env_var("YDC_API_KEY", strict=False)` | You.com API key. When unresolved, the keyless free tier is used. |
| `top_k` | `int \| None` | `10` | Max results per section (web, news), 1–100. Overridable per `run()`. |
| `freshness` | `str \| None` | `None` | `day`, `week`, `month`, `year`, or `YYYY-MM-DDtoYYYY-MM-DD`. |
| `country` | `str \| None` | `None` | 2-letter country code (e.g. `US`, `DE`). |
| `search_lang` | `str \| None` | `None` | Result language, BCP 47 (e.g. `EN`, `PT-BR`). |
| `safesearch` | `str \| None` | `None` | `off`, `moderate`, or `strict`. |
| `extra_params` | `dict \| None` | `None` | Extra query params passed through (e.g. `{"include_domains": "nytimes.com,bbc.com"}`). Comma-separated domain lists become JSON arrays on `POST`. |
| `timeout` | `int` | `10` | HTTP timeout in seconds. |
| `max_retries` | `int` | `3` | Retry attempts on transient failures. |
| `extraction_mode` | `str` | `"snippets"` | `snippets`, `highlights`, or `full_page`. Highlights and full page need an API key. |
| `extraction_formats` | `list[str] \| None` | `None` (API default `["markdown"]`) | Full-page formats: `markdown`, `html`, or both. |
| `crawl_timeout` | `int \| None` | `None` (API default 10) | Seconds (1-60) to wait for page content in `full_page` mode. |

`run(query, top_k=None)` returns `{"documents": list[Document], "links": list[str]}`. Web results come first, then news; each `Document` carries `title`, `url`, `source` (`web`/`news`), and `page_age` in its `meta`, with snippets (or the description) as `content` by default. With `extraction_mode` set to `highlights` or `full_page`, `meta` also carries `content_level` (and `content_format` for full pages).

`as_tool(name="youcom_web_search", description=...)` returns a `ComponentTool` for Haystack `Agent`; `documents_to_text(documents)` is the formatter it uses to render results for the model.

## Rate limits

Without an API key, requests use the keyless endpoint (currently 100 searches/day per IP, no livecrawl). When the limit is reached, the component raises an error explaining how to upgrade.

## Development

```bash
hatch run test:unit           # unit tests (mocked HTTP)
hatch run test:integration    # live API tests
hatch run fmt-check           # ruff lint + format check
hatch run test:types          # mypy
```

## License

Apache-2.0
