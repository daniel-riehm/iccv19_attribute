import sys

import torch
import cv2
import pandas as pd
from model.inception_iccv import inception_iccv
from torchvision import transforms
from utils.datasets import description


if __name__ == "__main__":
    model = inception_iccv()
    model.eval()
    model.cuda()

    video = cv2.VideoCapture(sys.argv[1])

    df = pd.read_csv(
        sys.argv[2],
        header=0,
        names=[
            "id",
            "file",
            "frame",
            "x0",
            "y0",
            "x1",
            "y1",
            "confidence",
            "target length",
            "class",
            "class confidence",
        ],
    )

    for label in description["rap"]:
        df.insert(len(df.columns), label, 0.0)

    df = df[df["confidence"] >= 0.8]
    df = df[(df["class"] == "person") | (df["class"] == "kitware-person")]

    frame_number = 1
    while True:
        have_img, frame_img = video.read()
        if not have_img:
            break
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        frame_df = df[(df["frame"] == frame_number)]
        indices_to_drop = []
        for row in frame_df.itertuples():
            img = frame_img[row.y0 : row.y1, row.x0 : row.x1]
            if not img.size:
                indices_to_drop.append(row.Index)
                continue
            img = transforms.ToTensor()(img).unsqueeze(0).cuda(non_blocking=True)
            img = transforms.Resize(size=(256, 128))(img)
            img = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(img)
            result = model.forward(img)
            result = torch.max(
                torch.max(torch.max(result[0], result[1]), result[2]), result[3]
            )
            result = torch.sigmoid(result.detach()).cpu().numpy().squeeze()
            for label, value in zip(description["rap"], map(float, result)):
                df.at[row.Index, label] = value
        df.drop(index=indices_to_drop, inplace=True)
        print(frame_number)
        frame_number += 1
    df.drop(
        columns=[
            "file",
            "x0",
            "y0",
            "x1",
            "y1",
            "confidence",
            "target length",
            "class",
            "class confidence",
        ],
        inplace=True,
    )
    df.to_csv(sys.argv[3], index=False, float_format="%.5f")
