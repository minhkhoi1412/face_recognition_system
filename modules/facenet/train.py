import os
import glob
import torch 
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import numpy as np
from utils import trans

IMG_PATH = './data/facenet'
MODEL_PATH = './models/facenet'
embeddings = []
names = []

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)
model.eval()

for usr in os.listdir(IMG_PATH):
    embeds = []
    for file in glob.glob(os.path.join(IMG_PATH, usr)+'/*.jpg'):
        try:
            img = Image.open(file)
        except:
            continue
        with torch.no_grad():
            embed = model(torch.unsqueeze(trans(img), 0).to(device))
            embeds.append(embed) # a picture size 1x512
    if len(embeds) == 0:
        continue
    embedding = torch.cat(embeds).mean(0, keepdim=True) # average of 50 pictures
    embeddings.append(embedding) # a list of N average of 50 pictures
    names.append(usr)

embeddings = torch.cat(embeddings) # [N, 512]
names = np.array(names)
torch.save(embeddings, MODEL_PATH + "/faceslist.pth")
np.save(MODEL_PATH + "/usernames", names)
