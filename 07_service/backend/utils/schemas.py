class QueryResult(BaseModel):
    image_ext_id: int
    item_ext_id: int
    similarity: float


class QueryRequest(BaseModel):
    text: str