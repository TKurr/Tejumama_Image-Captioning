import os
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score

def loadImage(image_path, target_size=(224, 224)):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        img = img.resize((target_size[1], target_size[0]))
        return np.asarray(img, dtype=np.float32) / 255.0

def loadBatch(image_paths, target_size=(224, 224)):
    if not image_paths:
        return np.empty((0, target_size[0], target_size[1], 3), dtype=np.float32)
    return np.stack([loadImage(p, target_size) for p in image_paths], axis=0)

def extractFeatures(
        image_paths,
        encoder,
        output_path,
        target_size=(224, 224),
        batch_size=32,
):
    features = []
    for start in range(0, len(image_paths), batch_size):
        batch = loadBatch(image_paths[start:start + batch_size], target_size)
        features.append(encoder.predict(batch, verbose=0))
    features = np.concatenate(features, axis=0) if features else np.empty((0, 0), dtype=np.float32)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, features)
    return features

def buildEncoder(input_size=(299, 299)):
    from tensorflow.keras.applications import InceptionV3
    
    model = InceptionV3(include_top=False, weights='imagenet', pooling='avg', input_shape=(*input_size, 3))
    model.trainable = False
    
    return model

def loadDataset(dataset_dir, class_names=None):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    if class_names is None:
        class_names = sorted(e for e in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, e)))
    image_paths, labels = [], []
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith(valid_exts):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(label)
    return image_paths, labels, class_names

def macroF1(y_true, y_pred):
    return float(f1_score(y_true, y_pred, average='macro'))
