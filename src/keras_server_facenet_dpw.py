# Python code for Keras Server

# Usage
# Starting the server:
# python keras_server.py
# 'http://localhost:5000/predict'
#  Client request can be given by:
#	python simple_request.py

# import the necessary packages
import os
import io
import requests
from flask import Flask, flash, redirect, render_template, request, session, abort, url_for, Response
import flask
from PIL import Image
import cv2
import pickle
import numpy as np
from collections import defaultdict
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from tabledef import *
from flask_session import Session
import torch
import torch.nn as nn
from torchvision import transforms
from modules.facenet.FaceReg import FaceReg
from modules.deep_pixel_wise.Model import DeePixBiS
from utility import img_to_encoding, resize_img
from facenet_pytorch import InceptionResnetV1, MTCNN


app = Flask(__name__)
app.secret_key = os.urandom(1)
engine = create_engine('sqlite:///login_db.db', echo=True)
model = None
user_db = None
IMAGE_SAVE_PATH = './images'

global device
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load facenet model
print('[INFO] Loading the Neural Network......\n')
recognizer = FaceReg(device=device)
print('[INFO] Model loaded..............')

# For detecting the face boundary


# Custom loss function for model


def face_present(image_path):
    img = cv2.imread(image_path, -1)
    save_loc = 'saved_image/new.jpg'
    face_present = False

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect face
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # required region for the face
        roi_color = img[y-90:y+h+70, x-50:x+w+50]

        # crop to 96 X 96, required by the model
        roi_color = cv2.resize(roi_color, (96, 96))
        # save the detected face
        cv2.imwrite(save_loc, roi_color)
        # make face present as true
        face_present = True

        # Just for visualization purpose
        # draw a rectangle bounding the face and save it
        roi = img[y-90:y+h+70, x-50:x+w+50]
        cv2.rectangle(img, (x-10, y-70),
                      (x+w+20, y+h+40), (15, 175, 61), 4)
        cv2.imwrite('static/saved_images/bounded.jpg', img)
    return face_present


# for loading the facenet trained model
def load_FRmodel():
    global model
    model = load_model('models/model_040922.h5',
                       custom_objects={'triplet_loss': triplet_loss})
    model.summary()


# load the saved user database
def ini_user_database():
    global user_db
    # check for existing database
    if os.path.exists('database/user_dict.pickle'):
        with open('database/user_dict.pickle', 'rb') as handle:
            user_db = pickle.load(handle)
    else:
        # make a new one
        # we use a dict for keeping track of mapping of each person with his/her face encoding
        user_db = defaultdict(dict)
    return user_db


# For adding user face
def add_face():
    data = {"face_present": False}
    encoding = None
    # CHECK FOR FACE IN THE IMAGE
    valid_face = False
    if valid_face := face_present('saved_image/new.jpg'):
        # create image encoding
        encoding = img_to_encoding('saved_image/new.jpg', model)
        # save the output for sending as json
        data['face_present'] = True
    else:
        # save the output for sending as json
        data['face_present'] = False
        print('No subject detected !')

    return data, encoding


# dashboard page
@app.route('/dashboard')
def dashboard():
    print(session.get('logged_in'))
    return flask.render_template('dashboard.html')


# index page
@app.route('/')
def index():
    return (
        dashboard()
        if session.get('logged_in')
        else flask.render_template("index.html")
    )


# login page
@app.route('/login')
def login():
    return flask.render_template("login.html")


# for verifying user
@app.route('/authenticate_user', methods=["POST"])
def authenticate_user():
    POST_USERNAME = str(request.form['exampleInputEmail1'])
    POST_PASSWORD = str(request.form['exampleInputPassword1'])
    # making a session
    Session = sessionmaker(bind=engine)
    s = Session()
    query = s.query(User).where(User.username.in_(
        [POST_USERNAME]), User.password.in_([POST_PASSWORD]))
    result = query.first()
    print(result)
    # if the user is logged in
    if result:
        session['logged_in'] = True
        return dashboard()
    else:
        flash('wrong password!')
    return login()


# logout page
@app.route("/logout", methods=['POST'])
def logout():
    # logging out the user
    flask.session['logged_in'] = False
    return index()


# Sign up page display
@app.route('/sign_up')
def sign_up():
    return flask.render_template("sign_up.html")


# to add user through the sign up from
@app.route('/signup_user', methods=["POST"])
def signup_user():
    # declaring the engine
    engine = create_engine('sqlite:///login_db.db', echo=True)

    # whether user registration was successful or not
    user_status = {'registration': False,
                   'face_present': False, 'duplicate': False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        print('Inside post')
        # getting the email and password from the user
        POST_USERNAME = str(request.form['email'])
        POST_PASSWORD = str(request.form['pass'])
        NAME = str(request.form['name'])

        if POST_USERNAME not in user_db.keys():
            # add new user's face
            if flask.request.files.get("image"):
                print('Inside Image')
                # read the image in PIL format
                image = flask.request.files["image"].read()
                image = np.array(Image.open(io.BytesIO(image)))
                print('Image saved success')
                # save the image on server side
                cv2.imwrite('saved_image/new.jpg',
                            cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                # check if any face is present or not in the picture
                data, encoding = add_face()
                # set face detected as True
                user_status['face_present'] = data['face_present']
            # if no image was sent
            else:
                user_status['face_present'] = False

            # only create a new session if complete user details is present
            if data['face_present']:
                # create a new session
                Session = sessionmaker(bind=engine)
                s = Session()
                # add data to user_db dict
                user_db[POST_USERNAME]['encoding'] = encoding
                user_db[POST_USERNAME]['name'] = NAME

                # save the user_db dict
                with open('database/user_dict.pickle', 'wb') as handle:
                    pickle.dump(user_db, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)
                print(f'User {POST_USERNAME} added successfully')

                # adding the user to data base
                user = User(POST_USERNAME, POST_PASSWORD)
                s.add(user)
                s.commit()

                # set registration status as True
                user_status['registration'] = True
                # logging in the user
                session['logged_in'] = True
                            # return dashboard()
        else:
            user_status['duplicate'] = True

    # return sign_up()
    return flask.jsonify(user_status)


def add_overlays_single_image(frame, faces):
    face_present = False
    authenticate = False
    name = 'Unknown Person'

    if faces is not None:
        face_present = True
        save_loc = 'saved_image/new.jpg'
        cv2.imwrite(save_loc, frame)

        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            cv2.imwrite('static/saved_images/bounded.jpg', frame)

            if face.name is not None and face.probability > 0.7:
                print(face.name)
                name = face.name
                authenticate = True
            else:
                authenticate = False

    return face_present, name, authenticate


def add_overlays_single_image_FAS(frame, faces):
    model = DeePixBiS()
    model.load_state_dict(torch.load(r'C:\KhoiNXM\Workspace\Learning\Master Thesis\Dev\face_recognition_system\models\DeePixBiS\DeePixBiS_celeb_nuaa_130223_v2.pth', 
                          map_location=torch.device(device)))
    model.eval()
    
    tfms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    face_present = False
    authenticate = False

    name = 'Unknown Person'
    fas_result = 'Fake Face'
    label = 0

    if faces is not None:
        face_present = True

        for face in faces:
            face_bb = face.bounding_box.astype(int)
            faceRegion = frame[face_bb[1]-20:face_bb[3]+20, face_bb[0]-20:face_bb[2]+20]
            faceRegion = cv2.cvtColor(faceRegion, cv2.COLOR_BGR2RGB)
            faceRegion = tfms(faceRegion)
            faceRegion = faceRegion.unsqueeze(0)
            mask, binary = model.forward(faceRegion)
            res = torch.mean(mask).item()

            label = 0 if res < 0.6 else 1
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            cv2.imwrite('./saved_images/cropped.png', frame)

            if face.name is not None and face.probability > 0.8 and label == 1:
                print(face.name)
                name = face.name
                authenticate = True
            else:
                authenticate = False

            fas_result = 'Real Face' if label == 1 else 'Fake Face'
            print(fas_result)

    return face_present, name, authenticate, fas_result, label


# predict function
@app.route("/predict", methods=["POST"])
def predict():
    # this will contain the
    data = {"success": False, 'authenticate': False}
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = np.array(Image.open(io.BytesIO(image)))

            # save the image on server side
            cv2.imwrite('./saved_image/user.png',
                        cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # CHECK FOR FACE IN THE IMAGE
            valid_face = False
            frame = cv2.imread('./saved_image/user.png', -1)
            faces = recognizer.identify(frame)
            valid_face, user_name, authenticate, fas_result, label = add_overlays_single_image_FAS(frame, faces)

            # do facial recognition only when there is a face inside the frame
            print(valid_face)
            if valid_face:
                # save the output for sending as json
                data['name'] = user_name if user_name != 'Unknown Person' else 'Unknown Person'
                data['spoof'] = label != 1
                data['face_present'] = True
                data['authenticate'] = authenticate

            else:
                # save the output for sending as json
                data['name'] = 'NaN'
                data['face_present'] = False
                data['authenticate'] = False
                print('No subject detected !')

            # indicate that the request was a success
            data["success"] = True

        # create a new session
        Session = sessionmaker(bind=engine)
        s = Session()
        # check if the user is logged in
        if data['authenticate']:
            session['logged_in'] = True
            session.modified = True
        else:
            flash('Unknown Person!')

    print(flask.jsonify(data))
    print(session.get('logged_in'))
    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# first load the model and then start the server
if __name__ == "__main__":

    print('[INFO] Starting Flask server.........Please wait until the server starts')
    # print('Loading the Neural Network......\n')
    # load_FRmodel()
    # print('Model loaded..............')
    ini_user_database()
    print('[INFO] Database loaded...........')
    app.run(host='127.0.0.1', port=5000)
