import os
import kagglehub
from pathlib import Path

DATA_DIR = Path(__file__).parent

def download():
    src = Path(kagglehub.dataset_download("puneet6060/intel-image-classification"))
    
    for folder in ('seg_train', 'seg_test', 'seg_pred'):
        dst = DATA_DIR / folder
        if dst.exists() or dst.is_symlink():
            print(f"skip {dst.name} (already exists)")
        else:
            os.symlink(src / folder, dst)
            print(f"linked {dst} -> {src / folder}")

if __name__ == '__main__':
    download()
