import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from flask import Flask, request, jsonify
import base64
import io
from PIL import Image
import pandas as pd
import zipfile
import numpy as np
import cv2
from flask_cors import CORS, cross_origin
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from base64 import b64decode
import random
from PIL import Image
from sklearn.model_selection import train_test_split


NUM_CLASSES = 7
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

class Classifier(nn.Module):
    def __init__(self, n_labels):
        super(Classifier, self).__init__()



        self.conv1 = nn.Conv2d(1, 128, 5)
        self.conv2 = nn.Conv2d(128, 64, 3)
        self.conv3 = nn.Conv2d(64, 32, 3)
        # self.conv4 = nn.Conv2d(32, 16, 3)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, NUM_CLASSES)


    def forward(self, x):
        shape = []
        for s in x.size():
          shape.append(s)
        shape.insert(1, 1)
        x = x.view(shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # print('l0', x.size())

        x = x.view(-1, self.flatten(x))
        # print('l1', x.size())

        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        # print('l2', x.size())
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def flatten(self, x):
        res = 1
        for s in x.size()[1:]:
            res *= s
        return res

def findEmotion(img):
  t = (torch.FloatTensor(img))
  t = t.view((1, 48, 48))
  t = torch.FloatTensor(t)
  device = classifier.conv1.weight.device
  t = t.to(device)
  testEmo = torch.argmax(classifier.forward(t))
  testEmo = testEmo.detach().cpu().numpy().tolist()
#   print("Predicted : ", mapper[testEmo])
  return mapper[testEmo]

def initSetup():
    # print("INIT called")
    device = torch.device("cpu")
    classifier = Classifier(NUM_CLASSES).to(device)
    classifier.load_state_dict(torch.load('classifierFinal.pt', map_location= torch.device('cpu')))
    classifier.eval()
    return classifier

classifier = initSetup()


class FaceDetector:
  def __init__(self):
    self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  def run(self):
    from PIL import Image, ImageOps
    with Image.open('output.jpg') as img:
      img = ImageOps.grayscale(img)
      img = np.array(img)
    face = self.face_classifier.detectMultiScale(
      img, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        break
    
    # plt.figure(figsize=(20,10))
    # plt.imshow(img, cmap='gray')
    # plt.axis('off')
    return self.extractSquare(face, img), face[0] if len(face) > 0 else [], img

  def extractSquare(self, face, img):
    # print(len(face), face)
    for (x, y, w, h) in face:
      cropped_image = img[y:y + h, x:x + w]
    #   plt.figure(figsize=(20,10))
    #   plt.imshow(cropped_image, cmap='gray')
    #   plt.axis('off')
      break
    return cropped_image


def parseImage(request):
    try:
        data = request.get_json()
        base64_image = data['image'].split(',')[1]
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        image.save('output.jpg')
        return None
        # return jsonify({'status': 'image saved'}), 200
    except Exception as e:
        # print(str(e))
        return str(e)


mapper = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
 }

def findEmotion(img):
  t = (torch.FloatTensor(img))
  t = t.view((1, 48, 48))
  t = torch.FloatTensor(t)
  device = classifier.conv1.weight.device
  t = t.to(device)
  testEmo = torch.argmax(classifier.forward(t))
  testEmo = testEmo.detach().cpu().numpy().tolist()
#   print("Predicted : ", mapper[testEmo])
  return mapper[testEmo]


@app.route('/upload', methods=['POST'])
def upload():
    try:
      faceDetector = FaceDetector()
      status = parseImage(request)
      if status != None:
        return status
      
      imgnp, rect, img = faceDetector.run()
      imgnp = cv2.resize(imgnp, (48, 48))

      x = base64.b64encode(img)
      # print(x[:10])
      encodedImg = x.decode('utf-8')
      temDec = encodedImg.encode('utf-8')
      print(temDec[:10])
      # print(encodedImg[:10])

      res = findEmotion(imgnp)
      jsonRes = jsonify({
          'emotion': res,
          'img': encodedImg,
          'rect': str(rect)
      })
      # print(jsonRes)
      return jsonRes, 200
    except Exception as e:
       return str(e)


    # return "ok"