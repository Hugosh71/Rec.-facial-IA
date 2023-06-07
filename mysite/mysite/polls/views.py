import base64
import io
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .camera import predict_emotion
from django.shortcuts import render
import json
import numpy as np
import cv2

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
