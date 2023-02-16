import cv2
import time
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from utils import load_faceslist, extract_face, inference

embeddings = []
names = []

device = 'cpu'
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

model = InceptionResnetV1(
    classify=False,
    pretrained="vggface2"
).to(device)
model.eval()

frame_interval = 3  # Number of frames after which to run face detection
fps_display_interval = 5  # seconds
frame_rate = 0
frame_count = 0

cap = cv2.VideoCapture(0)

mtcnn = MTCNN(thresholds=[0.7, 0.7, 0.8],
              post_process=False, keep_all=True, device=device)
embeddings, names = load_faceslist()

while cap.isOpened():
    isSuccess, frame = cap.read()
    start_time = time.time()

    if not isSuccess:
        continue

    start_det = time.time()
    boxes, _ = mtcnn.detect(frame)
    print(time.time() - start_det)

    if boxes is not None:
        boxes = boxes.astype('int').tolist()
        for bbox in boxes:
            if bbox[2] - bbox[0] <= 0 or bbox[3] - bbox[1] <= 0:
                continue
            try:
                face = extract_face(bbox, frame)
            except Exception:
                continue

            idx, score = inference(model, face, embeddings, names, device)
            score = torch.Tensor.cpu(score).detach().numpy()

            if score >= 0.7:
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 6)
                frame = cv2.putText(frame, names[idx] + '_{:.2f}'.format(score), 
                                    (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 2, cv2.LINE_8)
            else:
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)
                frame = cv2.putText(frame, 'Unknown', 
                                    (bbox[0], bbox[1]), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2, cv2.LINE_8)

    # Check our current fps
    end_time = time.time()
    # if (end_time - start_time) > fps_display_interval:
    #     print(frame_interval)
    frame_rate = int(1/(end_time - start_time))
    start_time = time.time()
    frame_count = 0

    cv2.putText(
        frame,
        f"{frame_rate} fps",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        thickness=2,
        lineType=2,
    )

    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
