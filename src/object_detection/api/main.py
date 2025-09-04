"""Simple FastAPI app for object detection."""

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import io
import sys
sys.path.append('src')

from object_detection.inference.predictor import ObjectDetector

# Create FastAPI app
app = FastAPI(title="Object Detection API", version="1.0.0")

# Initialize detector with your CPU config file
detector = ObjectDetector(config_path="configs/model/detection.yaml")

@app.get("/")
def home():
    """Welcome message."""
    return {"message": "Object Detection API is running!"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Upload an image and get detection results."""
    try:
        # Read uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Run detection
        results = detector.predict(image, save_results=False)
        
        # Return just the detections from first image
        detections = results[0]['detections'] if results else []
        
        return {
            "filename": file.filename,
            "detections": detections,
            "count": len(detections)
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Detection failed: {str(e)}"}
        )
