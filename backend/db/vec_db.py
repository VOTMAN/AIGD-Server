import os
import sys
from pathlib import Path

import chromadb

parent_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "app"
)
sys.path.append(parent_dir)

from embeddings import getImgEmbeddings, getTextEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VEC_DB_PATH = os.environ.get(
    "VEC_DB_PATH", os.path.join(BASE_DIR, "data", "chroma_vector_db")
)

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
VIDEO_DIR = os.path.join(DATA_DIR, "extractedFrames")
SAVE_DIR = Path(VIDEO_DIR) / "save"

client = chromadb.PersistentClient(path=VEC_DB_PATH)

collection = client.get_or_create_collection(
    name="clips",
    embedding_function=None,
    metadata={
        "description": "This collection is used to store the embeddings of a extracted clip with it id, prediction for the embedded frame, clip_name, and embedded frame's path"
    },
    configuration={"hnsw": {"space": "cosine", "ef_construction": 200}},
)

frames = [
    "/api/frames/80a4ed8a-29dc-49b9-9809-c8f473835a9c/frame_0004_medium.jpg",
    "/api/frames/80a4ed8a-29dc-49b9-9809-c8f473835a9c/frame_0015_medium.jpg",
    "/api/frames/80a4ed8a-29dc-49b9-9809-c8f473835a9c/frame_0011_medium.jpg",
]


def addEmbedding(backendFramePaths, clipName, prediction, id):
    print("vecdb")

    for i, f in enumerate(backendFramePaths):
        fid, filename = f.split("/")[3:]
        real_path = str(SAVE_DIR / fid / filename)

        emb = getImgEmbeddings(real_path)
        if emb is None:
            print(f"Failed embedding: {f}")
            continue

        collection.add(
            ids=[f"{clipName}_frame_{i:04d}"],
            embeddings=[emb.tolist()],
            documents=[clipName],
            metadatas=[
                {"path": [f], "game": prediction, "clip_name": clipName, "id": fid}
            ],
        )


def getEmbeddings(textQuery, n_results=20):

    queryEmbedding = getTextEmbeddings(textQuery)

    res = collection.query(
        query_embeddings=queryEmbedding.tolist(), n_results=n_results
    )

    clips = {}
    for meta, dist, doc in zip(
        res["metadatas"][0], res["distances"][0], res["documents"][0]
    ):
        clip_name = doc
        if clip_name not in clips or dist < clips[clip_name]["distance"]:
            clips[clip_name] = {"metadata": meta, "distance": dist}

    sorted_clips = sorted(clips.values(), key=lambda x: x["distance"])
    print({c["metadata"]["clip_name"]: c["distance"] for c in sorted_clips})
    return [[c["metadata"] for c in sorted_clips]]
