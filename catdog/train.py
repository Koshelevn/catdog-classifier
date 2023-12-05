import time

import numpy as np
import torch
import torch.nn as nn
from models.cnn import model_cnn
from sklearn.metrics import f1_score
from utils.data_loaders import train_dataset, val_dataset


NUM_WORKERS = 4
BATCH_SIZE = 256
EPOCH_NUM = 30

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def compute_loss(model, loss, data_batch):
    """Compute the loss using loss_function for the batch of data and return mean loss"""

    img_batch = data_batch["img"]
    label_batch = data_batch["label"]

    logits = model(img_batch)

    return loss(logits, label_batch), model


def accuracy(scores, labels, threshold=0.5):
    predicted = np.array(scores > threshold).astype(np.int32)
    return np.mean(predicted == labels)


def f1(scores, labels, threshold=0.5):
    predicted = np.array(scores > threshold).astype(np.int32)
    return f1_score(labels, predicted)


# you may add other metrics if you wish
tracked_metrics = {"accuracy": accuracy, "f1-score": f1}


def calculate_metrics(scores, labels, print_log=False):
    """Compute all the metrics from tracked_metrics dict using scores and labels."""

    scores_array = np.array(scores).astype(np.float32)
    labels_array = np.array(labels)

    metric_results = {}
    for k, v in tracked_metrics.items():
        metric_value = v(scores_array, labels_array)
        metric_results[k] = metric_value

    if print_log:
        print(" | ".join(["{}: {:.4f}".format(k, v) for k, v in metric_results.items()]))

    return metric_results


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


@torch.no_grad()  # we do not need to save gradients on evaluation
def test_model(model, loss, batch_generator, subset_name="test", print_log=True):
    """Evaluate the model using data from batch_generator and metrics defined above."""

    # disable dropout / use averages for batch_norm
    model.train(False)

    # save scores, labels and loss values for performance logging
    score_list = []
    label_list = []
    loss_list = []

    for X_batch, y_batch in batch_generator:
        # do the forward pass
        logits = model(X_batch.to(device))
        _, scores = torch.max(logits.cpu().data, 1)
        labels = y_batch.numpy().tolist()

        # compute loss value
        batch_loss = loss(logits, y_batch.to(device))

        # save the necessary data
        loss_list.append(batch_loss.detach().cpu().numpy().tolist())
        score_list.extend(scores)
        label_list.extend(labels)

    if print_log:
        print("Results on {} set | ".format(subset_name), end="")

    metric_results = calculate_metrics(score_list, label_list, print_log)
    metric_results["scores"] = score_list
    metric_results["labels"] = label_list
    metric_results["loss"] = loss_list

    return metric_results


def train_model(
    model,
    loss,
    train_batch_generator,
    val_batch_generator,
    opt,
    ckpt_name=None,
    n_epochs=EPOCH_NUM,
):
    """
    Run training: forward/backward pass using batch_generators.
    Log performance using loss monitoring and score distribution plots for validation set.
    """

    train_loss, val_loss = [], [1]
    val_loss_idx = [0]
    top_val_accuracy = 0

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train phase
        model.train(True)  # enable dropout / batch_norm training behavior
        for (X_batch, y_batch) in train_batch_generator:
            # move data to target device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            data_batch = {"img": X_batch, "label": y_batch}
            batch_loss, model = compute_loss(model, loss, data_batch)

            # compute backward pass
            opt.zero_grad()
            batch_loss.backward()
            opt.step()

            # log train loss
            train_loss.append(batch_loss.detach().cpu().numpy())

        # Evaluation phase
        metric_results = test_model(model, loss, val_batch_generator, subset_name="val")
        metric_results = get_score_distributions(metric_results)

        # Logging
        val_loss_value = np.mean(metric_results["loss"])
        val_loss_idx.append(len(train_loss))
        val_loss.append(val_loss_value)

        print(
            "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, n_epochs, time.time() - start_time
            )
        )
        val_accuracy_value = metric_results["accuracy"]
        if val_accuracy_value > top_val_accuracy and ckpt_name is not None:
            top_val_accuracy = val_accuracy_value

            # save checkpoint of the best model
            with open(ckpt_name, "wb") as f:
                torch.save(model, f)
    print(f"Accuracy for saved model: {top_val_accuracy}")
    return model


def train_and_save_model(model, model_name):
    """Train chosen model and save it"""

    train_batch_gen = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_batch_gen = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    opt.zero_grad()
    ckpt_name = model_name
    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    train_model(
        model, loss, train_batch_gen, val_batch_gen, opt, ckpt_name=ckpt_name, n_epochs=10
    )


if __name__ == "__main__":
    train_and_save_model(model_cnn, "cnn.ckpt")
