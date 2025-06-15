from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient

from utils.vector_db import VectorDBClient
from utils.routes import tfidf_search, clip_search
from utils.schemas import QueryRequest, QueryResult

db_client = QdrantClient(url="http://localhost:6333")
vdb_tfidf = VectorDBClient(db_client, "tfidf")
vdb = VectorDBClient(db_client, "clip_openai")


app = FastAPI()


@app.post("/search-tfidf/", response_model=list[QueryResult])
async def search_tfidf(request: QueryRequest):
    results: list[QueryResult] = tfidf_search(request.text, vdb_tfidf)
    return results



@app.post("/search/", response_model=list[QueryResult])
async def search_clip(request: QueryRequest):
    results = clip_search(request.text)
    return results