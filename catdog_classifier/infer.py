from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imread
from skimage.transform import resize
from utils.dvc_helpers import dvc_pull


@hydra.main(config_path="config", config_name="conf", version_base="1.1")
def infer_model(cfg):
    """Run model on selected image"""

    dvc_pull(cfg.data.model_dir)

    model_dir = Path(cfg.data.model_dir)
    with open(model_dir / cfg.model.filename, "rb") as file:
        model = torch.load(file)

    src = imread(cfg.infer.file)
    resized = resize(src, (cfg.data.size_height, cfg.data.size_width), mode="reflect")

    tensor = torch.Tensor(
        np.transpose(
            (resized / 255 - cfg.data.image_mean) / cfg.data.image_std, [2, 0, 1]
        )[np.newaxis, :, :, :]
    ).to(torch.device("cpu"))
    model.eval()
    score_cat = F.softmax(model(tensor), 1)[0][0].detach().cpu().numpy()
    score_dog = F.softmax(model(tensor), 1)[0][1].detach().cpu().numpy()
    print(f"cat score: {score_cat}\ndog score: {score_dog}")


if __name__ == "__main__":
    infer_model()
