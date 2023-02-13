import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss
from facenet_pytorch import InceptionResnetV1, MTCNN
import os


model = DeePixBiS()
model.load_state_dict(torch.load(
    r'C:\KhoiNXM\Workspace\Learning\Master Thesis\Dev\face_recognition_system\models\DeePixBiS\DeePixBiS_celeb_nuaa_130223.pth', 
    map_location=torch.device('cuda'))
)

model.eval()

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

mtcnn = MTCNN(thresholds=[0.5, 0.7, 0.9],
              post_process=False, keep_all=True, device='cuda')

camera = cv.VideoCapture(0)
while cv.waitKey(1) & 0xFF != ord('q'):
    _, img = camera.read()
    boxes, _ = mtcnn.detect(img)

    if boxes is not None:
        boxes = boxes.astype('int').tolist()

        for bbox in boxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            if w - x <= 0 or h - y <= 0:
                continue

            try:
                faceRegion = img[y-20:h+20, x-20:w+20]
                faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
                # print(os.getcwd())
                # cv.imwrite('./test.png', faceRegion)
                # cv.imshow('Test', faceRegion)

                faceRegion = tfms(faceRegion)
                faceRegion = faceRegion.unsqueeze(0)

                # mask, binary = model.forward(faceRegion)
                mask, binary = model.forward(faceRegion)
                res = torch.mean(mask).item()
                # res = binary.item()
                print(res)

                if res < 0.6:
                    cv.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
                    cv.putText(img, 'Fake_{:.2f}'.format(res), (x, y - 30),
                               cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
                else:
                    cv.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
                    cv.putText(img, 'Real_{:.2f}'.format(res), (x, y - 30),
                               cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))
            except Exception:
                continue

        cv.imshow('Deep Pixel-wise Binary Supervision Anti-Spoofing', img)

camera.release()
cv.destroyAllWindows()
