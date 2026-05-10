import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from sklearn.metrics import f1_score

# IMAGE UTILITIES
def __loadimage__(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((target_size[1], target_size[0]))
        return np.asarray(img, dtype=np.float32) / 255.0


def __loadbatch__(
    image_paths: List[str],
    target_size: Tuple[int, int] = (224, 224)
) -> np.ndarray:
    if not image_paths:
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32)

    images = [__loadimage__(path, target_size) for path in image_paths]
    return np.stack(images, axis=0).astype(np.float32)


def __extractfeatures__(
    image_paths: List[str],
    encoder,                         # Keras Model (frozen CNN encoder)
    output_path: str,
    target_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32
) -> np.ndarray:
    features = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        batch = __loadbatch__(batch_paths, target_size)
        features.append(encoder.predict(batch, verbose=0))

    if features:
        features = np.concatenate(features, axis=0)
    else:
        features = np.empty((0, 0), dtype=np.float32)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, features)
    return features


# DATASET UTILITIES
def __loaddataset__(
    dataset_dir: str,
    class_names: Optional[List[str]] = None
) -> Tuple[List[str], List[int], List[str]]:
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    if class_names is None:
        class_names = sorted(
            entry for entry in os.listdir(dataset_dir)
            if os.path.isdir(os.path.join(dataset_dir, entry))
        )

    image_paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file_name in sorted(os.listdir(class_dir)):
            if file_name.lower().endswith(valid_exts):
                image_paths.append(os.path.join(class_dir, file_name))
                labels.append(label)

    return image_paths, labels, class_names


def __macrof1__(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))
