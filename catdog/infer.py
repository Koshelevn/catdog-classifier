import numpy as np
import torch
import torch.nn.functional as F
from skimage.io import imread
from skimage.transform import resize


with open("cnn.ckpt", "rb") as file:
    model = torch.load(file)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

SIZE_H = SIZE_W = 96
image_mean = [0.485, 0.456, 0.406]
image_std = [0.229, 0.224, 0.225]


def infer_model(model, img_path):
    """Run model on selected image"""

    src = imread(img_path)

    resized = resize(src, (SIZE_H, SIZE_W), mode="reflect")
    # convert to torch.Tensor
    tensor = torch.Tensor(
        np.transpose((resized / 255 - image_mean) / image_std, [2, 0, 1])[
            np.newaxis, :, :, :
        ]
    ).to(device)

    print(F.softmax(model(tensor), 1))
    score_cat = F.softmax(model(tensor), 1)[0][0].detach().cpu().numpy()
    score_dog = F.softmax(model(tensor), 1)[0][1].detach().cpu().numpy()
    return score_cat, score_dog
