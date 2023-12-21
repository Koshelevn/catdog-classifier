import mlflow


def get_experiment(name="catdog_train"):
    existing_exp = mlflow.get_experiment_by_name(name)
    if not existing_exp:
        experiment_id = mlflow.create_experiment(name)
        mlflow.set_experiment(name)
    else:
        current_experiment = dict(existing_exp)
        experiment_id = current_experiment["experiment_id"]
    return experiment_id
