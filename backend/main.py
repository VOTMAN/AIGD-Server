import asyncio
import os
import re
import shutil
import sys
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta


from fastapi import FastAPI, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.status import HTTP_504_GATEWAY_TIMEOUT

from db.models import PredResults
from db.db import initDB, saveResult, getResult, getAllResults

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app"))
sys.path.append(parent_dir)

from utils import detectFrame, detectVideo, loadReferenceEmbeddings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.environ.get("DATA_DIR", "/data")
VIDEO_DIR = os.path.join(DATA_DIR, "extractedFrames")
SAVE_DIR = Path(VIDEO_DIR) / "save"

CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(",")

TIME_REGX = r"^(?:[0-5]\d):(?:[0-5]\d)$"

REQUEST_TIMEOUT_ERROR = int(os.environ.get("REQUEST_TIMEOUT", "120"))

referenceEmbeddings = loadReferenceEmbeddings()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def cleanupOldFrames():
    while True:
        await asyncio.sleep(60 * 60)
        now = datetime.now()
        for folder in os.listdir(VIDEO_DIR):
            folder_path = os.path.join(VIDEO_DIR, folder)
            age = datetime.fromtimestamp(os.path.getmtime(folder_path))
            if now - age > timedelta(hours=2):
                shutil.rmtree(folder_path)


@app.on_event("startup")
async def startup():
    initDB()
    asyncio.create_task(cleanupOldFrames())


@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_ERROR)

    except asyncio.TimeoutError:
        process_time = time.time() - start_time
        return JSONResponse(
            {
                "detail": "Request processing time excedeed limit",
                "processing_time": process_time,
            },
            status_code=HTTP_504_GATEWAY_TIMEOUT,
        )


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.get("/api/results/all")
async def getAllRes():
    res: list[PredResults] = getAllResults()
    return res


@app.get("/api/results/{id}")
async def getRes(id: str):
    res: PredResults = getResult(id)
    return Response(
        content=res.model_dump_json(),
        media_type="application/json",
        headers={"Cache-Control": "public, max-age=3600"}
    )


@app.get("/api/frames/{id}/{filename}")
async def get_frame(id: str, filename: str):

    requested_path = (SAVE_DIR / id / filename).resolve()

    # Ensure requested path stays inside SAVE_DIR
    if not str(requested_path).startswith(str(SAVE_DIR.resolve())):
        return {"error": "Invalid path"}

    # Ensure file exists
    if not requested_path.exists():
        return {"error": "Frame not found, Rerun Detection"}

    return FileResponse(
        requested_path, 
        media_type="image/jpeg", 
        headers={
        "Cache-Control": "public, max-age=86400"
        }
    )


@app.post("/api/upload/clip")
async def predClip(
    file: UploadFile,
    startTime: str | None = Form("00:00"),
    endTime: str | None = Form(None),
):
    temp_id = str(uuid.uuid4())
    temp_video_path = f"/tmp/{temp_id}"
    img_save_dir = f"{SAVE_DIR}/{temp_id}"

    if not re.fullmatch(TIME_REGX, startTime):
        return {"error": "Invalid startTime format. Use MM:SS"}

    if endTime == "":
        endTime = None

    if endTime is not None and not re.fullmatch(TIME_REGX, endTime):
        return {"error": "Invalid endTime format. Use MM:SS"}

    if startTime == endTime:
        print("Error: Start time and end time cannot be equal")
        return {"error": "Start and end times are equal"}

    with open(temp_video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # print(temp_video_path, startTime, endTime)
        result = detectVideo(temp_video_path, referenceEmbeddings, startTime, endTime)

        if result is None:
            return {"error": "Could not detect game"}

        base_parent_dir = os.path.dirname(
            os.path.dirname(result["influential_frames"][0])
        )
        os.makedirs(img_save_dir, exist_ok=True)

        for path in result["influential_frames"]:
            print("Img Path: ", path)
            base_name: str = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[
                0
            ]
            med_img_name = base_name + "_medium.jpg"
            shutil.copy2(
                os.path.join(base_parent_dir, "medium", med_img_name), img_save_dir
            )

        res = saveResult(PredResults(
            id=temp_id,
            clip_name=file.filename,
            prediction=result["prediction"],
            confidences=result["confidences"],
            frames=[f"/api/frames/{temp_id}/{f}" for f in os.listdir(img_save_dir)],
            time_taken=result["time_taken"]
        ))

        print(res)
        return {"id": temp_id}

    except Exception as e:
        print(e)
        return {"error": str(e)}

    finally:
        os.remove(temp_video_path)
        extract_dir = os.path.join(VIDEO_DIR, temp_id)
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)


@app.post("/api/upload/frame")
async def predFrame(file: UploadFile):
    temp_id = str(uuid.uuid4())
    temp_frame_path = f"/tmp/{temp_id}"

    with open(temp_frame_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = detectFrame(temp_frame_path, referenceEmbeddings)
        return {"prediction": result[0], "confidence": result[1]}
    except Exception as e:
        print(e)
        return {"error": str(e)}
    finally:
        os.remove(temp_frame_path)
