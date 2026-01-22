"""
Inference module for face mask detection using YOLOv8
"""
import os
import logging
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "model/best.pt"
DEFAULT_DEVICE = "cpu"


def load_model(model_path: str = MODEL_PATH, device: str = DEFAULT_DEVICE) -> Tuple[Optional[YOLO], Optional[str]]:
    """
    Load and cache the YOLO model
    
    Args:
        model_path: Path to the model file
        device: Device to run inference on ('cpu', 'cuda', 'mps')
    
    Returns:
        Tuple of (model, error_message)
        - model: Loaded YOLO model or None if failed
        - error_message: Error description or None if successful
    """
    try:
        # Check if model file exists
        if not os.path.exists(model_path):
            error_msg = f"Model file not found at: {model_path}"
            logger.error(error_msg)
            return None, error_msg
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"Loading model from {model_path} ({file_size:.2f} MB)")
        
        # Load model
        model = YOLO(model_path)
        model.to(device)
        
        # Log model info
        logger.info(f"Model loaded successfully on {device}")
        logger.info(f"Model classes: {model.names}")
        
        return model, None
        
    except FileNotFoundError as e:
        error_msg = f"Model file not found: {str(e)}"
        logger.error(error_msg)
        return None, error_msg
    
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        logger.error(error_msg)
        return None, error_msg

### Video processing

def run_inference_on_frame(model, frame, conf):
    results = model(frame, conf=conf)[0]

    annotated = frame.copy()

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        confidence = float(box.conf[0])

        label = f"{model.names[cls_id]} {confidence:.2f}"
        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return annotated


def run_inference(
    model: YOLO,
    image: Image.Image,
    conf: float = 0.5,
    iou: float = 0.45,
    verbose: bool = False
):
    """
    Run YOLOv8 inference on a PIL image
    
    Args:
        model: Loaded YOLO model
        image: PIL Image to run inference on
        conf: Confidence threshold for detections
        iou: IoU threshold for NMS
        verbose: Whether to print verbose output
    
    Returns:
        YOLO Results object containing detections
    
    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If inference fails
    """
    try:
        # Validate inputs
        if model is None:
            raise ValueError("Model is None. Please load a valid model first.")
        
        if not isinstance(image, Image.Image):
            raise ValueError("Image must be a PIL Image object")
        
        if not 0 <= conf <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {conf}")
        
        if not 0 <= iou <= 1:
            raise ValueError(f"IoU must be between 0 and 1, got {iou}")
        
        # Convert PIL to OpenCV format
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        logger.info(f"Running inference with conf={conf}, iou={iou}")
        
        # Run inference
        results = model.predict(
            img_bgr,
            conf=conf,
            iou=iou,
            verbose=verbose
        )
        
        # Log detection results
        if results and len(results) > 0:
            num_detections = len(results[0].boxes)
            logger.info(f"Detected {num_detections} objects")
        
        return results
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    
    except Exception as e:
        error_msg = f"Inference failed: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def preprocess_image(image: Image.Image, target_size: Optional[Tuple[int, int]] = None) -> Image.Image:
    """
    Preprocess image before inference
    
    Args:
        image: Input PIL Image
        target_size: Optional tuple (width, height) to resize to
    
    Returns:
        Preprocessed PIL Image
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            logger.debug("Converted image to RGB")
        
        # Resize if target size is specified
        if target_size:
            image = image.resize(target_size, Image.LANCZOS)
            logger.debug(f"Resized image to {target_size}")
        
        return image
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise


def get_model_info(model: YOLO) -> dict:
    """
    Get information about the loaded model
    
    Args:
        model: Loaded YOLO model
    
    Returns:
        Dictionary containing model information
    """
    try:
        info = {
            'model_type': model.model.__class__.__name__,
            'device': str(model.device),
            'class_names': model.names,
            'num_classes': len(model.names),
        }
        return info
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {}