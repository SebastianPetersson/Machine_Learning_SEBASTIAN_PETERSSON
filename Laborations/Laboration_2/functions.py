import pandas as pd
import matplotlib.pyplot as plt
import json
from torchvision.io import decode_image
from torchvision.transforms.v2.functional import to_pil_image
from torchcam.methods import LayerCAM
from torchcam.utils import overlay_mask
import torch

COLUMNS = [
    "filename",
    "class_target",
    "role",
    "image_description",
    "source_url",
    "source_site",
    "license",
    "notes",
]

def create_image_sources():
    """Create an empyt DataFrame for image source metadata."""
    return pd.DataFrame(columns=COLUMNS)

def add_image_sources(
    image_sources,
    filename,
    class_target,
    role,
    image_description,
    source_url="",
    source_site="",
    license="",
    notes=""
):
    """Add one image entry to the image sources DataFrame."""
    new_row = pd.DataFrame([{
        "filename": filename,
        "class_target": class_target,
        "role": role,
        "image_description": image_description,
        "source_url": source_url,
        "source_site": source_site,
        "license": license,
        "notes": notes,
    }])

    return pd.concat([image_sources, new_row], ignore_index=True)

def save_image_sources(image_sources, path):
    """Save the image sources DataFrame to a CSV file."""
    image_sources.to_csv(path, index=False)

def predict_class(output_tensor: torch.Tensor, class_index_path: str) -> dict: # From Claude
    """
    Maps the maximum logit in a softmax output tensor to an ImageNet class name.
    
    Args:
        output_tensor:  1D or 2D tensor of shape (1000,) or (1, 1000),
                        typically the output of a softmax layer from ResNet18.
        
        class_index_path: Path to the imagenet_class_index.json file.
        
    Returns:
        A dict with keys:
            - 'class_index' (int)   : index of the predicted class (0-999)
            - 'class_id'    (str)   : WordNet synset ID, e.g. "n01140764"
            - 'class_name'  (str)   : human-readable label, e.g. "tench"
            - 'confidence'  (float) : softmax probability of the top class
            """
    
    with open(class_index_path, "r") as f:
        class_index = json.load(f)  # keys are str "0".."999"

        # Flatten to 1-D in case the tensor has a batch dimension
        probs = output_tensor.squeeze()     #(1000,)
        if probs.ndim != 1 or probs.shape[0] != 1000:
            raise ValueError(
                f"Expected a tensor of 1000 values, got shape {tuple(output_tensor.shape)}"
            )
        
        top_idx = int(probs.argmax())           # index of highest probability
        synset_id, class_name = class_index[str(top_idx)]

        return {
            "class_index":  top_idx,
            "class_id":     synset_id,
            "class_name":   class_name,
            "confidence":   float(probs[top_idx])
        }

def load_and_preprocess_image(image_path, preprocess):
    """Loads an image and applies preprocessing."""
    img = decode_image(str(image_path))
    input_tensor = preprocess(img)
    return img, input_tensor

def get_prediction(model, input_tensor):
    """Runs the model and returns raw output and softmax prediction."""
    model.eval()
    output = model(input_tensor.unsqueeze(0))
    prediction = output.squeeze(0).softmax(0)
    return output, prediction

def generate_cam(model, input_tensor, target_layer=None):
    """Generates a LayerCAM activation map for an input image."""
    with LayerCAM(model, target_layer=target_layer) as cam_extractor:
        output = model(input_tensor.unsqueeze(0))
        class_idx = output.squeeze(0).argmax().item()
        activation_map = cam_extractor(class_idx, output)
    return output, class_idx, activation_map


def plot_cam_results(img, activation_map):
    """Plot the activation map and the CAM overlay."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(activation_map[0].squeeze(0).numpy())
    plt.axis("off")
    plt.title("Activation map")

    result = overlay_mask(
        to_pil_image(img),
        to_pil_image(activation_map[0].squeeze(0), mode="F"),
        alpha=0.5
    )
    
    plt.subplot(1, 2, 2)
    plt.imshow(result)
    plt.axis("off")
    plt.title("CAM overlay")
    plt.tight_layout()
    plt.show()

def run_analysis_pipeline(image_path, model, preprocess, class_index_path, target_layer=None):
    "Run the full prediction and CAM analysis pipeline for one image."
    img, input_tensor = load_and_preprocess_image(image_path, preprocess)

    output, prediction = get_prediction(model, input_tensor)
    _, class_idx, activation_map = generate_cam(model, input_tensor, target_layer=target_layer)

    predicted_class = predict_class(prediction.detach(), class_index_path)

    print("Predicted class:", predicted_class["class_name"])
    print("Confidence:", predicted_class["confidence"])
    print("Target layer:", target_layer if target_layer else "default")

    plot_cam_results(img, activation_map)

    return {
        "image_path": str(image_path),
        "output": output,
        "predicted_class": predicted_class,
        "class_idx": class_idx,
        "activation_map": activation_map,
        "prediction": prediction,
        "target_layer": target_layer
    }


def top_k_predictions(prediction, class_index_path, k=5):
    """Returns the top-k predicted classes from a softmax prediction."""
    with open(class_index_path, "r", encoding="utf-8") as f:
        class_index = json.load(f)

    probs = prediction.squeeze()
    top_probs, top_indices = torch.topk(probs, k)

    results = []

    for idx, prob in zip(top_indices.tolist(), top_probs.tolist()):
        synset_id, class_name = class_index[str(idx)]
        results.append({
            "class_index": idx,
            "class_id": synset_id,
            "class_name": class_name,
            "confidence": prob
        })

    return pd.DataFrame(results)


def top_k_logits(output, class_index_path, k=5):
    """Return the top-k classes based on raw model logits."""
    with open(class_index_path, "r", encoding="utf-8") as f:
        class_index = json.load(f)

    logits = output.squeeze()
    top_logits, top_indices = torch.topk(logits, k)

    results = []

    for idx, logit in zip(top_indices.tolist(), top_logits.tolist()):
        synset_id, class_name = class_index[str(idx)]
        results.append({
            "class_index": idx,
            "class_id": synset_id,
            "class_name": class_name,
            "logit": logit
        })

    return pd.DataFrame(results)
