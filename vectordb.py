
import asyncio
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance,Filter,VectorParams,models,ScoredPoint


client=AsyncQdrantClient(url="http://localhost:6333")
COLLECTION_NAME="memeries"

class EmbeddedMemory(BaseModel):
    user_id:int
    memory_text:str
    embedding:list[float]

class RetrievedMemory(BaseModel):
    point_id:str
    user_id:int
    memory_text:str
    score:float


async def create_memory_collection():
    if not (await client.collection_exists(COLLECTION_NAME)):
        await client.create_collection(COLLECTION_NAME,vectors_config=VectorParams(size=64,distance=Distance.DOT))
        await client.create_payload_index(collection_name=COLLECTION_NAME,field_name="user_id",field_schema=models.PayloadSchemaType.INTEGER)
        print("Collection Created")
    else:
        print("Collection already exist")


async def insert_memories(memories:list[EmbeddedMemory]):
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=uuid4().hex,
                payload={
                    "user_id":memory.user_id,
                    "memory_text":memory.memory_text,
                },
                vector=memory.embedding
            )
            for memory in memories
        ]
    )

async def search_memories(search_vector:list[float],user_id:int,limit:Optional[int]=2):
    must_conditions:list[models.Condition]=[models.FieldCondition(key="user_id",match=models.MatchValue(value=user_id))]

    outs=await client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        with_payload=True,
        query_filter=Filter(must=must_conditions),
        score_threshold=0.1,
        limit=limit,
    )
    return [
        convert_retrived_records(point) for point in outs.points if outs.points is not None
    ]

async def delete_records(point_ids:list[str]):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids)
    )


def stringify_retrieved_point(retrieved_memory: RetrievedMemory):
    return f"""{retrieved_memory.memory_text} Relevance: {retrieved_memory.score:.2f}"""

def convert_retrived_records(point)->RetrievedMemory:
    return RetrievedMemory(
        point_id=point.id,
        user_id=point.payload["user_id"],
        memeory_text=point.payload["memory_text"],
        score=point.score
    )

if __name__=="__main__":
    asyncio.run(create_memory_collection())




