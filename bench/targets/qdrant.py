"""Qdrant 1.16+ target — ACORN native, external SOTA baseline."""

import numpy as np
import httpx


class QdrantTarget:
    name = "qdrant"
    COLLECTION = "bench_items"

    def __init__(self, base_url: str = "http://localhost:6333"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(base_url=self.base_url, timeout=60.0)
        self.ef_search = None   # None = Qdrant server default (~128)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def set_ef_search(self, n: int) -> None:
        """Runtime ef knob — passed per-request as params.hnsw_ef."""
        self.ef_search = n

    def _search_params(self) -> dict:
        return {"params": {"hnsw_ef": self.ef_search}} if self.ef_search else {}

    def setup(self, vectors: np.ndarray, metadata: list[dict]) -> None:
        dim = vectors.shape[1]

        # drop if exists
        self.client.delete(f"/collections/{self.COLLECTION}")

        # Align HNSW build params with pgvector/ACORN (m=16, ef_construct=64)
        # for a fair cross-engine comparison.
        self.client.put(
            f"/collections/{self.COLLECTION}",
            json={
                "vectors": {"size": dim, "distance": "Cosine"},
                "hnsw_config": {"m": 16, "ef_construct": 64},
            },
        ).raise_for_status()

        # create payload index for bucket field (enables server-side filtering)
        self.client.put(
            f"/collections/{self.COLLECTION}/index",
            json={"field_name": "bucket", "field_schema": "integer"},
        ).raise_for_status()

        # batch upsert
        points = [
            {
                "id": i + 1,
                "vector": v.tolist(),
                "payload": {"bucket": m["bucket"]},
            }
            for i, (v, m) in enumerate(zip(vectors, metadata))
        ]
        self.client.put(
            f"/collections/{self.COLLECTION}/points",
            json={"points": points},
        ).raise_for_status()

    def query_filtered(self, query: np.ndarray, bucket_threshold: int, k: int) -> list[int]:
        resp = self.client.post(
            f"/collections/{self.COLLECTION}/points/search",
            json={
                "vector": query.tolist(),
                "limit": k,
                "filter": {
                    "must": [
                        {"key": "bucket", "range": {"lt": bucket_threshold}}
                    ]
                },
                **self._search_params(),
            },
        )
        resp.raise_for_status()
        return [hit["id"] for hit in resp.json()["result"]]

    def query_unfiltered(self, query: np.ndarray, k: int) -> list[int]:
        resp = self.client.post(
            f"/collections/{self.COLLECTION}/points/search",
            json={"vector": query.tolist(), "limit": k, **self._search_params()},
        )
        resp.raise_for_status()
        return [hit["id"] for hit in resp.json()["result"]]

    def insert_batch(self, _vectors: np.ndarray, _metadata: list[dict]) -> None:
        # Qdrant requires stable IDs across upserts; incremental scenario
        # must use setup() with the full dataset instead.
        raise NotImplementedError("use setup() with full dataset for Qdrant")

    def teardown(self) -> None:
        self.client.delete(f"/collections/{self.COLLECTION}")

    def close(self) -> None:
        self.client.close()
