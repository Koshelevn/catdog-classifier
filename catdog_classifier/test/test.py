import torch

from catdog_classifier.test.metrics import calculate_metrics


@torch.no_grad()
def test_model(model, loss, batch_generator, subset_name="test", print_log=True):
    """Evaluate the model using data from batch_generator and metrics defined above."""

    model.train(False)

    score_list = []
    label_list = []
    loss_list = []

    for X_batch, y_batch in batch_generator:

        logits = model(X_batch.to(torch.device("cpu")))
        _, scores = torch.max(logits.cpu().data, 1)
        labels = y_batch.numpy().tolist()

        batch_loss = loss(logits, y_batch.to(torch.device("cpu")))

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
