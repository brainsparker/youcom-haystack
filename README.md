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
| `extra_params` | `dict \| None` | `None` | Extra query params passed through (e.g. `{"include_domains": "nytimes.com,bbc.com"}`). |
| `timeout` | `int` | `10` | HTTP timeout in seconds. |
| `max_retries` | `int` | `3` | Retry attempts on transient failures. |

`run(query, top_k=None)` returns `{"documents": list[Document], "links": list[str]}`. Web results come first, then news; each `Document` carries `title`, `url`, `source` (`web`/`news`), and `page_age` in its `meta`, with snippets (or the description) as `content`.

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
