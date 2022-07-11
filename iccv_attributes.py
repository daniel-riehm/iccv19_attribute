import sys
import torch
import json
import cv2
import pandas as pd
from model.inception_iccv import inception_iccv
from torchvision import transforms
from utils.datasets import description
from pprint import pprint
from time import time


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

    results = []
    frame_number = 0
    while True:
        have_img, frame_img = video.read()
        if not have_img:
            break
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        frame_df = df[(df["frame"] == frame_number)]
        for row in frame_df.itertuples(index=False):
            if row.confidence < 0.5:
                continue
            img = frame_img[row.y0 : row.y1, row.x0 : row.x1]
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
            result = dict(zip(description["rap"], map(float, result)))
            results.append({"frame": frame_number, "iccv19_attribute_rap": result})
        print(frame_number, time())
        frame_number += 1
    with open(sys.argv[3]) as f:
        json.dump(results, f)
