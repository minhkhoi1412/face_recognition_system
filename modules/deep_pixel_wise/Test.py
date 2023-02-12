import cv2 as cv
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from Model import DeePixBiS
from Loss import PixWiseBCELoss
from Metrics import predict, test_accuracy, test_loss


model = DeePixBiS()
model.load_state_dict(torch.load(
    r'C:\KhoiNXM\Workspace\Learning\Master Thesis\Reference Sources\Face-Anti-Spoofing-DeePixBiS\DeePixBiS_celeb_nuaa_041222.pth', 
    map_location=torch.device('cpu')))
model.eval()

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

faceClassifier = cv.CascadeClassifier(r'C:\KhoiNXM\Workspace\Learning\Master Thesis\Dev\face_recognition_system\models\haar\haarface.xml')

camera = cv.VideoCapture(0)

# width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
# height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
# size = (width, height)
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('output_DeePixBiS.mp4', fourcc, 5.0, size)

while cv.waitKey(1) & 0xFF != ord('q'):
    _, img = camera.read()
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = faceClassifier.detectMultiScale(
        grey, scaleFactor=1.1, minNeighbors=4)

    if faces is not None:
        for x, y, w, h in faces:
            faceRegion = img[y:y+h, x:x+w]
            faceRegion = cv.cvtColor(faceRegion, cv.COLOR_BGR2RGB)
            cv.imwrite('./test.png', faceRegion)
            # cv.imshow('Test', faceRegion)

            faceRegion = tfms(faceRegion)
            faceRegion = faceRegion.unsqueeze(0)

            mask, binary = model.forward(faceRegion)
            res = torch.mean(mask).item()
            # res = binary.item()
            print(res)

            # cv.putText(img, 'Confidence_{:.2f}'.format(res),
            #            (x, y), cv.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv.LINE_8)

            if res < 0.6:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(img, 'Fake_{:.2f}'.format(res), (x+w//3, y + h + 30),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255))
            else:
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(img, 'Real_{:.2f}'.format(res), (x+w//3, y + h + 30),
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0))

        # out.write(img)
        cv.imshow('Deep Pixel-wise Binary Supervision Anti-Spoofing', img)

camera.release()
# out.release()
cv.destroyAllWindows()
