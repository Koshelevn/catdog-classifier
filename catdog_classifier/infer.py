from pathlib import Path

import hydra
import pandas as pd
import torch
import torch.nn as nn
from utils.data_loaders import get_datasets
from utils.dvc_helpers import dvc_pull

from catdog_classifier.test.test import test_model


@hydra.main(config_path="config", config_name="conf", version_base="1.1")
def infer_model(cfg):
    """Run model on test dataset"""

    dvc_pull(cfg.data.model_dir)

    _, _, test_dataset = get_datasets(
        cfg.data.paths.folder,
        cfg.data.paths.train,
        cfg.data.paths.val,
        cfg.data.paths.test,
        cfg.data.size_height,
        cfg.data.size_width,
        cfg.data.image_mean,
        cfg.data.image_std,
    )
    test_batch_gen = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.model.train.batch,
        num_workers=cfg.model.train.workers,
    )

    model_dir = Path(cfg.data.model_dir)

    loss = nn.CrossEntropyLoss()
    with open(model_dir / cfg.model.filename, "rb") as file:
        model = torch.load(file)

    metric_results = test_model(model, loss, test_batch_gen, subset_name="test")
    files = list(map(lambda x: x[0].split("/")[-1], test_dataset.imgs))
    result_df = pd.DataFrame(
        {
            "filename": files,
            "predict": metric_results.get("labels"),
            "category": metric_results.get("labels"),
        }
    )
    result_df.category.replace({0: "cat", 1: "dog"}, inplace=True)
    path = Path(cfg.data.inference_result)
    result_df.to_csv(path, index=False)


if __name__ == "__main__":
    infer_model()
