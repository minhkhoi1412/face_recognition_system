import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization

MODEL_PATH = './models/facenet'

def trans(img):
    transform = transforms.Compose([
            transforms.ToTensor(),
            # fixed_image_standardization
        ])
    return transform(img)

def load_faceslist():
    embeds = torch.load(MODEL_PATH+'/faceslist.pth')
    names = np.load(MODEL_PATH+'/usernames.npy')
    return embeds, names


def extract_face(box, img):
    img = img[box[1]:box[3], box[0]:box[2]]
    face = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
    face = Image.fromarray(face)
    return face


def inference(model, face, local_embeds, names, device):
    embed = model(torch.unsqueeze(trans(face), 0).to(device))
    norm_score = torch.nn.functional.cosine_similarity(embed, local_embeds)
    min_dist, embed_idx = torch.max(norm_score, dim=0)
    print(min_dist, names[embed_idx])
    return embed_idx, min_dist.double()
