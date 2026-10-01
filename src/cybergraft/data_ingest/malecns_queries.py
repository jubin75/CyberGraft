"""Typed query wrappers around neuPrint for the MaleCNS feeding pathway."""

from datetime import datetime, timezone
from typing import Any

from .cache import QueryCache
from .neuprint_client import NeuPrintClient

_NEURON_FIELDS = ("bodyId", "type", "instance", "superclass", "class", "predictedNt", "status")


class MaleCNSQueries:
    def __init__(
        self,
        client,
        cache: QueryCache,
        *,
        dataset: str,
        source_url: str,
        batch_size: int = 200,
    ) -> None:
        self.client = client
        self.cache = cache
        self.dataset = dataset
        self.source_url = source_url
        self.batch_size = batch_size
        self._query_log: list[dict] = []

    @property
    def query_log(self) -> list[dict]:
        return self._query_log

    def _neuron_subset(self, rows: list[dict]) -> list[dict]:
        return [{field: row.get(field) for field in _NEURON_FIELDS} for row in rows]

    def _query(self, query: dict, fetcher) -> list[dict]:
        key = self.cache.query_hash(query)
        cached = self.cache.get(key)
        if cached is not None:
            self._query_log.append({"hash": key, "query": query, "hit_cache": True})
            return cached
        result = fetcher()
        self.cache.put(key, result, source_url=self.source_url, dataset=self.dataset, query=query)
        self._query_log.append({"hash": key, "query": query, "hit_cache": False})
        return result

    def resolve_neurons_by_type(self, types: list[str]) -> list[dict]:
        criteria = {"type": types, "status": "Traced"}
        query = {"op": "resolve_neurons", "criteria": criteria}
        return self._query(query, lambda: self._neuron_subset(self.client.query_neurons(criteria)))

    def resolve_neurons_by_body_ids(self, body_ids: list[int]) -> list[dict]:
        if not body_ids:
            return []
        rows: list[dict] = []
        for chunk in NeuPrintClient.chunked(body_ids):
            criteria = {"bodyId": list(chunk)}
            query = {"op": "resolve_neurons", "criteria": criteria}
            rows.extend(
                self._query(query, lambda c=criteria: self._neuron_subset(self.client.query_neurons(c)))
            )
        return rows

    def fetch_connections(self, sources, targets, min_weight: int = 1) -> list[dict]:
        query = {"op": "connections", "sources": sources, "targets": targets, "min_weight": min_weight}
        return self._query(query, lambda: self.client.query_connections(sources, targets, min_weight))

    def query_vpn_candidates(self) -> dict:
        query = {"op": "vpn_candidates"}
        criteria = {"superclass": ["visual_projection", "visual_projection_tbc"], "status": "Traced"}
        rows = self._query(query, lambda: self._neuron_subset(self.client.query_neurons(criteria)))
        return {
            "candidate_type": "visual_projection",
            "count": len(rows),
            "type_count": len({r["type"] for r in rows}),
            "dataset": self.dataset,
            "query": criteria,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
