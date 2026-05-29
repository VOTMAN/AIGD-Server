import os
from pathlib import Path
import sys
import chromadb

parent_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app")
sys.path.append(parent_dir)

from embeddings import buildClipEmbeddings, getTextEmbeddings

BASE_DIR = (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEC_DB_PATH = os.environ.get("VEC_DB_PATH", os.path.join(BASE_DIR, "data", "chroma_vector_db")) 

DATA_DIR = os.environ.get(
    "DATA_DIR",
    os.path.join(BASE_DIR, "data")
)
VIDEO_DIR = os.path.join(DATA_DIR, "extractedFrames")
SAVE_DIR = Path(VIDEO_DIR) / "save"

client = chromadb.PersistentClient(path=VEC_DB_PATH)

collection = client.get_or_create_collection(
    name="clips", embedding_function=None,
    metadata={
        "description": "This collection is used to store the embeddings of a extracted clip with it id, prediction for the embedded frame, clip_name, and embedded frame's path"
    },
    configuration={
        "hnsw": {
            "space": "cosine",
            "ef_construction": 200
        }
    }
)

frames= [
    "/api/frames/80a4ed8a-29dc-49b9-9809-c8f473835a9c/frame_0004_medium.jpg",
    "/api/frames/80a4ed8a-29dc-49b9-9809-c8f473835a9c/frame_0015_medium.jpg",
    "/api/frames/80a4ed8a-29dc-49b9-9809-c8f473835a9c/frame_0011_medium.jpg"
  ]

def addEmbedding(backendFramePaths, clipName, prediction, id):
    real_frames = []
    # print(backendFramePaths)
    print("vecdb")

    for f in backendFramePaths:
        id, filename = f.split("/")[3:]
        real_path = str(SAVE_DIR / id / filename)
        real_frames.append(real_path)


    clipEmbeddings = buildClipEmbeddings(real_frames)

    if clipEmbeddings == None:
        print("No embeddings returned builder")
        return

    collection.add(
        ids=[clipName],
        embeddings=[clipEmbeddings.tolist()],
        documents=[clipName],
        metadatas=[{"path": backendFramePaths, "game": prediction, "clip_name": clipName, "id": id}]
    )

    # print(collection.peek(limit=5))

def getEmbeddings(textQuery):

    queryEmbedding = getTextEmbeddings(textQuery)

    res = collection.query(
        query_embeddings=queryEmbedding.tolist(),
        n_results=5
    )
    return res["metadatas"]
