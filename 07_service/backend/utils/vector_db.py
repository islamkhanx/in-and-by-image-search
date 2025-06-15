from typing import Any, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class VectorDBClient:
    def __init__(self, db_client: QdrantClient, collection_name: str):
        self.db_client = db_client
        self.collection_name = collection_name
        if not db_client.collection_exists(collection_name):
            db_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )

    def find(self, vector: list[float]) -> Optional[Dict[str, Any]]:
        results = self.db_client.query_points(
            collection_name=self.collection_name,
            query=vector,
            with_payload=True,
            limit=10
        ).points

        return [result.model_dump() for result in results]

    def insert(self, vector: list[float], payload: Dict[str, Any], point_id: str) -> None:
        point = PointStruct(id=point_id, vector=vector, payload=payload)

        self.db_client.upsert(
            self.collection_name, [point], wait=False
        )

    def insert_multiple(self, vectors: list[list[float]], payloads: list[Dict[str, Any]], point_ids: list[str]) -> None:
        points = [
            PointStruct(id=point_id, vector=vector, payload=payload)
            for vector, payload, point_id in zip(vectors, payloads, point_ids)
        ]

        self.db_client.upsert(
            self.collection_name, points, wait=False
        )
