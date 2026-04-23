from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import cv2

# ------------------- Preprocessing for Classification -------------------
def preprocess_classification(image: Image.Image, image_size=(224, 224)) -> torch.Tensor:
    """
    Transform PIL image to tensor suitable for classification model.
    - image: PIL Image
    - image_size: target size (default 224x224 for ResNet)
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return tensor


# ------------------- Preprocessing for Detection/Segmentation -------------------
def preprocess_cv_image(file_bytes: bytes) -> np.ndarray:
    """
    Convert raw bytes to OpenCV numpy array (BGR format).
    """
    try:
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image data")
        return img
    except Exception as e:
        raise RuntimeError(f"Error preprocessing CV image: {e}")


# ------------------- Postprocessing -------------------
def get_top_predictions(output_tensor: torch.Tensor, class_labels: list) -> dict:
    """
    Convert classification model outputs to readable predictions.
    - output_tensor: raw model output
    - class_labels: list of class names
    """
    try:
        probs = torch.nn.functional.softmax(output_tensor, dim=1)
        confidence, pred_idx = torch.max(probs, 1)

        return {
            "predicted_class": class_labels[pred_idx.item()],
            "confidence": float(confidence.item()),
            "all_predictions": [
                {"class": class_labels[i], "confidence": float(p.item())}
                for i, p in enumerate(probs[0])
            ]
        }
    except Exception as e:
        raise RuntimeError(f"Error in postprocessing predictions: {e}")


def convert_masks_to_list(masks_tensor: torch.Tensor) -> list:
    """
    Convert YOLO segmentation masks tensor to list of lists.
    """
    try:
        masks_list = []
        for mask in masks_tensor:
            masks_list.append(mask.cpu().numpy().tolist())
        return masks_list
    except Exception as e:
        raise RuntimeError(f"Error converting masks: {e}")