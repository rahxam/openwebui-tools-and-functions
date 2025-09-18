"""
title: Azure AI Search
author: secondhandrebel
url: https://github.com/alecgraham/openwebui.functions
version: 0.1.0
license: MIT
requirements: azure-search, azure-search-documents
"""

from pydantic import BaseModel, Field


from typing import Awaitable, Callable, Dict, List, Any, Optional, Iterable
import json

# from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import VectorizableTextQuery
from azure.core.credentials import AzureKeyCredential


class EventEmitter:
    """Helper class to emit events to the UI"""

    def __init__(self, event_emitter: Callable[[dict], Awaitable[None]]):
        self.event_emitter = event_emitter
        pass

    async def emit_status(self, description: str, done: bool, error: bool = False):
        await self.event_emitter(
            {
                "data": {
                    "description": f"{done and (error and '❌' or '✅') or '🔎'} {description}",
                    "status": done and "complete" or "in_progress",
                    "done": done,
                },
                "type": "status",
            }
        )

    async def emit_message(self, content: str):
        await self.event_emitter({"data": {"content": content}, "type": "message"})

    async def emit_source(self, name: str, url: str, content: str, html: bool = False):
        await self.event_emitter(
            {
                "type": "citation",
                "data": {
                    "document": [content],
                    "metadata": [{"source": url, "html": html}],
                    "source": {"name": name},
                },
            }
        )


class Tools:
    class Valves(BaseModel):
        SEARCH_INDEX: str = Field(default="", description="Azure AI Search Index")
        SEARCH_API_KEY: str = Field(default="", description="Azure AI Search API Key")
        SEARCH_SERVICE: str = Field(
            default="", description="Azure AI Search Service Name"
        )
        RESULTS_NO: str = Field(default="4", description="Number of results to return")

    # Add your custom tools using pure Python code here, make sure to add type hints
    # Use Sphinx-style docstrings to document your tools, they will be used for generating tools specifications
    # Please refer to function_calling_filter_pipeline.py file from pipelines project for an example

    def __init__(self):
        print(f"[AI Search] initializing search")
        self.valves = self.Valves()
        self.user_valves = None

    # Add a description of contents of the knowledgebase
    async def azure_ai_search(
        self,
        search_query: str,
        __event_emitter__: Callable[[dict], Awaitable[None]],
        __user__: dict = {},
    ) -> str:
        """
        This search tool will search an internal knowledgebase.  You must formulate your own search query based on the user's message.
        :param search_query: A semantic search query used in search engine.
        :return: The search results in as json
        """

        event_emitter = EventEmitter(__event_emitter__)
        try:
            # Do not include :param for __user__ in the docstring as it should not be shown in the tool's specification
            # The session user object will be passed as a parameter when the function is called
            print("[AI Search] Executing AI Search")
            print(f"[AI Search] search_query: {search_query}")
            await event_emitter.emit_status(
                f"Searching Sharepoint for '{search_query}'...", False
            )
            # azd_credential = DefaultAzureCredential()
            azd_credential = AzureKeyCredential(self.valves.SEARCH_API_KEY)

            search_client = SearchClient(
                endpoint=f"https://{self.valves.SEARCH_SERVICE}.search.windows.net/",
                index_name=self.valves.SEARCH_INDEX,
                credential=azd_credential,
            )
            vector_query = VectorizableTextQuery(
                text=search_query,
                kind="text",
                k_nearest_neighbors=self.valves.RESULTS_NO,
                fields="contentVector",
            )

            search_results = search_client.search(
                search_text=search_query,
                top=self.valves.RESULTS_NO,
                include_total_count=True,
                query_type="semantic",
                semantic_configuration_name="default",
                vector_queries=[vector_query],
            )
            print(f"[AI Search] search complete.")
            await event_emitter.emit_status(f"Done retrieving Sharepoint results", True)
            results = []
            try:
                for r in search_results:
                    result = {
                        "id": r["id"],
                        "title": r["metadata_spo_item_name"],
                        "body": r["content"],
                        "source": r["metadata_spo_item_path"],
                    }
                    print(result)
                    await event_emitter.emit_source(
                        result["title"], result["source"], result["body"]
                    )
                    results.append(result)
            except Exception as e:
                print(e)

            return json.dumps(results)
        except Exception as e:
            await event_emitter.emit_status(
                f"Unexpected error during search: {str(e)}.", True, True
            )
            return f"Error: {str(e)}"
