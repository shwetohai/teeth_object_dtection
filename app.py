from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from inference import get_model
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = get_model(model_id="teeth_annotation_sl_techno/1")


# DTO for the image request
class TeethRequestDto(BaseModel):
    teeth_image: str


# DTO for the response of the process-image endpoint
class TeethResponseDto(BaseModel):
    x_min: List[float]
    y_min: List[float]
    width: List[float]
    height: List[float]
    teeth_number: List[int]
    success: bool
    error: Optional[str]
    results: Optional[List[Dict[str, Any]]] = []


# Initialize router
router = APIRouter()


@router.post("/process-image/", response_model=TeethResponseDto)
async def process_image(dto: TeethRequestDto) -> TeethResponseDto:
    try:
        # Perform inference
        result = model.infer(image=dto.teeth_image, confidence=0.2)

        # Extract predictions
        predictions = result[0].predictions
        x_min = []
        y_min = []
        width = []
        height = []
        teeth_number = []
        results = []

        # Iterate through detections
        for pred in predictions:
            x_min.append(pred.x - pred.width / 2)
            y_min.append(pred.y - pred.height / 2)
            width.append(pred.width)
            height.append(pred.height)
            teeth_number.append(pred.class_id)
            results.append(
                {
                    "x_min": float(pred.x - float(pred.width / 2)),
                    "y_min": float(pred.y - float(pred.height / 2)),
                    "width": float(pred.width),
                    "height": float(pred.height),
                }
            )

        return TeethResponseDto(
            x_min=x_min,
            y_min=y_min,
            width=width,
            height=height,
            teeth_number=teeth_number,
            success=True,
            error="",
            results=results,
        )
    except Exception as e:
        return TeethResponseDto(
            x_min=[],
            y_min=[],
            width=[],
            height=[],
            teeth_number=[],
            success=False,
            error=str(e),
            results=[],
        )


# Include the router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8111)