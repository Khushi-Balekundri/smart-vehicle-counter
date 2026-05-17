from fastapi import FastAPI, UploadFile, Form
from pydantic import BaseModel
from typing import Optional, Tuple, List
import shutil
import json
from tracker import process_video

app = FastAPI()

@app.post("/process")
async def process_video_api(
    file: UploadFile,
    mode: str = Form(...),
    model: str = Form("yolov11s"),
    zone: str = Form(None),
    line: str = Form(None),
):
    # save uploaded file
    input_path = f"videos/{file.filename}"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    model_path = f"models/{model}.pt"

    # parse coordinates from JSON strings
    zone_coords = tuple(json.loads(zone)) if zone else None
    line_coords = [tuple(p) for p in json.loads(line)] if line else None

    result = process_video(
        input_path = input_path,
        mode       = mode,
        zone       = zone_coords,
        line       = line_coords,
        model_path = model_path
    )

    return {
        "status":      "processed",
        "count":       result["count"],
        "output_path": result["output_path"]
    }