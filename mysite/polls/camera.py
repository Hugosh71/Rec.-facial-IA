import base64
import io
import json
import cv2
import numpy as np
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from deepface import DeepFace
from torch import nn
import torch.nn.functional as F
import torch

def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    ]

    if pool:
        layers.append(nn.MaxPool2d(kernel_size=2))

    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, num_classes=7, in_channels=1, lr=0.01, dropout=0.5, num_hidden=4096, model_name="ResNet9"):
        super(ResNet9, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.lr = lr
        self.dropout = dropout
        self.num_hidden = num_hidden
        self.model_name = model_name
        self.conv1 = conv_block(in_channels, 16, pool=False)  # 16 x 48 x 48
        self.conv2 = conv_block(16, 32, pool=True)  # 32 x 24 x 24
        self.res1 = nn.Sequential(  #  32 x 24 x 24
            conv_block(32, 32, pool=False),
            conv_block(32, 32, pool=False)
        )

        self.conv3 = conv_block(32, 64, pool=True)  # 64 x 12 x 12

        self.res2 = nn.Sequential(  # 128 x 6 x 6
            conv_block(64, 64),
            conv_block(64, 64)
        )

        self.conv4 = conv_block(64, 128, pool=True)  # 128 x 6 x 6

        self.res3 = nn.Sequential(  # 128 x 6 x 6
            conv_block(128, 128),
            conv_block(128, 128)
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),  # 128 x 3 x 3
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, self.num_hidden),  # 512
            nn.Linear(self.num_hidden, num_classes)  # 7
        )
        self.network = nn.Sequential(
            self.conv1,
            self.conv2,
            self.res1,
            self.conv3,
            self.res2,
            self.conv4,
            self.res3,
            self.classifier,
        )

    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.res2(out) + out
        out = self.conv4(out)
        out = self.res3(out) + out
        out = self.classifier(out)
        return out


def ResNet_function(img):
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model_trained = torch.load("C:\\Users\\fanch\\OneDrive\\Bureau\\mysite\\polls\\model.pth")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    
    model = ResNet9(num_classes=7, in_channels=1, lr=model_trained['lr'], dropout=model_trained['dropout'],
                    num_hidden=model_trained['num_hidden'], model_name="ResNet9")
    model.load_state_dict(model_trained['model_state_dict'])
    model.eval()

    
    for face in faces:
        x, y, w, h = face

        
        face_img = gray[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (48, 48))

        
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=0)

        with torch.no_grad():
            output = model(torch.tensor(face_img))
            predicted_emotion = emotion_labels[torch.argmax(output).item()]

        print(predicted_emotion)

        return predicted_emotion

    return "Dont Know"


def predict_emotion(img, method):
    if method == "ResNet":
        return ResNet_function(img)
    elif method == "DeepFace":
        detected_emotions = DeepFace.analyze(img, actions=['emotion'])
        first_result = detected_emotions[0]
        predicted_emotion = max(first_result['emotion'], key=first_result['emotion'].get)
        return predicted_emotion
    else:
        return "None"




@csrf_exempt
def emotion_detection(request):
    emotion = {'emotion': 'None'}
    if request.method == 'POST':
        print('POST request received')
        data = json.loads(request.body)
        image_data = data['image']
        image_data = base64.b64decode(image_data.split(',')[1])
        image_data = np.frombuffer(image_data, dtype=np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        if 'method' in data:
            method = data['method']
        else:
            method = "ResNet"
        print(method)
        emotion['emotion'] = predict_emotion(image, method)
        return JsonResponse(emotion)
    return render(request, 'index.html', emotion)
