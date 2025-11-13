import sys
import numpy as np
import pandas as pd
import kaggle
import os 
from typing import Any, Callable, Optional

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import decode_image

kaggle.api.authenticate()
kaggle.api.dataset_download_files('ananthu017/emotion-detection-fer', path='.',unzip=True)
kaggle.api.dataset_metadata('ananthu017/emotion-detection-fer', path='.')
#print(sys.path)


#ok here we go


SEARCH_ROOT_DIR = "train" #this is the image folder we're using


def find_png_files() -> list[str]:
    """Find all png files in the search root directory and its subdirectories."""
    png_file_paths: list[str] = []
    for current_dir, _subdirs, files in os.walk(SEARCH_ROOT_DIR):
        for file_name in files:
            if file_name.lower().endswith(".png"):
                png_file_paths.append(os.path.join(current_dir, file_name))
    return png_file_paths


def generate_dataframe(png_file_paths: list[str]) -> pd.DataFrame:
    """Generate a DataFrame with image paths and labels from JPG file paths."""
    data = []
    for image_path in png_file_paths:
        label = os.path.basename(os.path.dirname(image_path))
        data.append({"image_path": image_path, "label": label})
    return pd.DataFrame(data, columns=["image_path", "label"])


if __name__ == "__main__":
    png_file_paths = find_png_files()
    df = generate_dataframe(png_file_paths)
    print(df.head())
    df.to_csv("annotations.csv", index=False)

#ok other part:

class CustomImageDataset(Dataset):
    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.img_labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Any]:
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


if __name__ == "__main__":
    dataset = CustomImageDataset("annotations.csv", ".")
    print(dataset[0])
    print(dataset[1])

def main(): 

    print("hello world")

    

if __name__=="__main__": #this is very important 
    main()
    #I guess this is how we are going to start.
    #use python -m venv myproject-env for compiling code
    # then env\Scripts\activate to run
    #to kill env type deactivate
    #print("hello world")
    