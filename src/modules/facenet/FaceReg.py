import cv2
import time
import torch
from PIL import Image
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
from modules.facenet.utils import trans

DATA_PATH = r'C:\KhoiNXM\Workspace\Learning\Master Thesis\Dev\face_recognition_system\models\facenet'


class Face():
    def __init__(self, name, bounding_box, probability):
        self.name = name
        self.bounding_box = bounding_box
        self.probability = probability


class FaceReg():
    def __init__(self, device):
        self.device = device
        self.model = InceptionResnetV1(
            classify=False, pretrained="vggface2").to(self.device)
        self.model.eval()
        self.mtcnn = MTCNN(thresholds=[0.5, 0.7, 0.9], 
                           post_process=False, keep_all=True, device=self.device)
        self.load_faceslist()

    def load_faceslist(self):
        self.local_embeds = torch.load(f'{DATA_PATH}/faceslist.pth')
        self.local_names = np.load(f'{DATA_PATH}/usernames.npy')

    def inference(self, face):
        embedding = self.model(torch.unsqueeze(trans(face), 0).to(self.device))
        norm_score = torch.nn.functional.cosine_similarity(
            embedding, self.local_embeds)
        min_dist, embed_idx = torch.max(norm_score, dim=0)
        name = self.local_names[embed_idx]
        return name, min_dist

    def extract_face(self, box, img):
        img = img[box[1]:box[3], box[0]:box[2]]
        face = cv2.resize(img, (160, 160), interpolation=cv2.INTER_AREA)
        face = Image.fromarray(face)
        return face

    def identify(self, frame):
        boxes, _ = self.mtcnn.detect(frame)
        faces = []

        if boxes is not None:
            print("Face Detected")
            boxes = boxes.astype('int').tolist()
            spoofs = []
            for bbox in boxes:
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                if w - x <= 0 or h - y <= 0:
                    continue
                try:
                    face = self.extract_face(bbox, frame)
                except Exception:
                    continue
                name, probability = self.inference(face)
                faces.append(Face(name, np.array(bbox), probability))

        return faces or None
