import subprocess


def dvc_pull(path):
    subprocess.run(["dvc", "pull", path, "--force"])
