import sys
import glob
import json

sys.path.insert(0, 'src/wajib')

from cnn.utils import buildEncoder, extractFeatures

IMAGES_DIR  = 'data/flickr8k/Images/*.jpg'
OUTPUT_NPY  = 'src/wajib/weights/features/flickr8k_features.npy'
OUTPUT_IDX  = 'src/wajib/weights/features/flickr8k_index.json'


image_paths = sorted(glob.glob(IMAGES_DIR))
names       = [p.split('/')[-1] for p in image_paths]

print(f"Found {len(image_paths)} images")
encoder = buildEncoder()
extractFeatures(
    image_paths,
    encoder,
    output_path=OUTPUT_NPY,
    target_size=(299, 299),
    batch_size=32,
)

with open(OUTPUT_IDX, 'w') as f:
    json.dump(names, f)

print(f"Saved features -> {OUTPUT_NPY}")
print(f"Saved index    -> {OUTPUT_IDX}")
