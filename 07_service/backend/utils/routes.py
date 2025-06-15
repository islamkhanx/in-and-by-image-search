from model import get_text_vector, get_tfidf_vector, get_image_vector
from schemas import QueryResult
from vector_db import VectorDBClient


def tfidf_search(query: str, vdb_client: VectorDBClient) -> list[QueryResult]:
    vector = get_tfidf_vector(query)
    results = vdb_client.find(vector)

    return [
        QueryResult(
            image_ext_id=result["id"],
            item_ext_id=result["payload"]["item_ext_id"],
            similarity=result["score"]
        )
        for result in results
    ]

def clip_search(query: str, vdb_client: VectorDBClient) -> list[QueryResult]:
    vector = get_text_vector(query)
    results = vdb_client.find(vector)

    return [
        QueryResult(
            image_ext_id=result["id"],
            item_ext_id=result["payload"]["item_ext_id"],
            similarity=result["score"]
        )
        for result in results
    ]