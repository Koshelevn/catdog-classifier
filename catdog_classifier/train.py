import os
import time
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
import torch.nn as nn
from hydra import utils
from models.cnn import ModelCNN
from utils.data_loaders import get_datasets
from utils.dvc_helpers import dvc_pull
from utils.mlflow_helpers import get_experiment

from catdog_classifier.test.test import test_model


def compute_loss(model, loss, data_batch):
    """Compute the loss using loss_function for the batch of data and return mean loss"""

    img_batch = data_batch["img"]
    label_batch = data_batch["label"]

    logits = model(img_batch)

    return loss(logits, label_batch), model


def get_score_distributions(epoch_result_dict):
    """Return per-class score arrays."""
    scores = epoch_result_dict["scores"]
    labels = epoch_result_dict["labels"]

    # save per-class scores
    for class_id in [0, 1]:
        epoch_result_dict["scores_" + str(class_id)] = np.array(scores)[
            np.array(labels) == class_id
        ]

    return epoch_result_dict


def train_model(
    model,
    loss,
    train_batch_generator,
    val_batch_generator,
    opt,
    model_folder,
    ckpt_name,
    n_epochs,
):
    """
    Run training: forward/backward pass using batch_generators.
    Log performance using loss monitoring and score distribution plots for validation set.
    """

    top_val_accuracy = 0
    model_folder = Path(model_folder)

    for epoch in range(n_epochs):
        start_time = time.time()

        model.train(True)
        for (X_batch, y_batch) in train_batch_generator:

            X_batch, y_batch = X_batch.to(torch.device("cpu")), y_batch.to(
                torch.device("cpu")
            )
            data_batch = {"img": X_batch, "label": y_batch}
            batch_loss, model = compute_loss(model, loss, data_batch)

            opt.zero_grad()
            batch_loss.backward()
            opt.step()

        metric_results = test_model(model, loss, val_batch_generator, subset_name="val")
        metric_train = test_model(model, loss, train_batch_generator, subset_name="train")
        metric_results = get_score_distributions(metric_results)

        print(
            "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time
            )
        )
        val_accuracy_value = metric_results["accuracy"]
        mlflow.log_metric("val_accuracy", val_accuracy_value)
        mlflow.log_metric("val_f1", metric_results["f1-score"])
        mlflow.log_metric("train_accuracy", metric_train["accuracy"])
        mlflow.log_metric("train_f1", metric_train["f1-score"])
        if val_accuracy_value > top_val_accuracy and ckpt_name is not None:
            top_val_accuracy = val_accuracy_value
            if not os.path.exists(model_folder):
                os.mkdir(model_folder)

            with open(model_folder / ckpt_name, "wb") as f:
                torch.save(model, f)
    print(f"Accuracy for saved model: {top_val_accuracy}")
    return model


@hydra.main(config_path="config", config_name="conf", version_base="1.1")
def train_and_save_model(cfg):
    """Train chosen model and save it"""

    dvc_pull(cfg.data.paths.folder)

    train_dataset, val_dataset, _ = get_datasets(
        cfg.data.paths.folder,
        cfg.data.paths.train,
        cfg.data.paths.val,
        cfg.data.paths.test,
        cfg.data.size_height,
        cfg.data.size_width,
        cfg.data.image_mean,
        cfg.data.image_std,
    )
    train_batch_gen = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.model.train.batch,
        shuffle=True,
        num_workers=cfg.model.train.workers,
    )
    val_batch_gen = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.model.train.batch, num_workers=cfg.model.train.workers
    )
    model = ModelCNN(cfg.model.embedding_size).get_model()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.model.train.learning_rate)
    opt.zero_grad()
    model = model.to(torch.device("cpu"))
    loss = nn.CrossEntropyLoss()

    mlflow.set_tracking_uri("file://" + utils.get_original_cwd() + "/mlruns")
    experiment_id = get_experiment("catdog_train")
    with mlflow.start_run(experiment_id=experiment_id):
        train_model(
            model,
            loss,
            train_batch_gen,
            val_batch_gen,
            opt,
            model_folder=cfg.data.model_dir,
            ckpt_name=cfg.model.filename,
            n_epochs=cfg.model.train.epoch,
        )


if __name__ == "__main__":
    train_and_save_model()
