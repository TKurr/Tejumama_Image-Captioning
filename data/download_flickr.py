import os
import kagglehub
from pathlib import Path

DATA_DIR = Path(__file__).parent

def download():
    src = Path(kagglehub.dataset_download("adityajn105/flickr8k"))  
    dst = DATA_DIR / 'flickr8k'
    if dst.exists() or dst.is_symlink():
        print(f"skip flickr8k (already exists)")
    else:
        os.symlink(src, dst)
        print(f"linked {dst} -> {src}")

if __name__ == '__main__':
    download()
