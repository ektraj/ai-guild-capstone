

import numpy as np
import os
#import tensorflow as tf
from numpy.random import seed
from collections import deque
#from tensorflow import set_random_seed
import cv2
import random
import numpy as np
seed=1
random.seed(seed)
np.random.seed(seed)
#tf.random.set_seed(seed)
from collections import deque
import streamlit as st
import tempfile

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

cudnn.benchmark = True
plt.ion()   # interactive mode
from PIL import Image


# In[ ]:

modelfile = st.file_uploader("Upload model")
videofile = st.file_uploader("Upload video")
st.write(modelfile)



if modelfile is not None and videofile  is not None:
    mfile = tempfile.NamedTemporaryFile(delete=False)
    mfile.write(modelfile.read())
    
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(videofile.read())
    
    model = torch.load(mfile.name)
    CLASSES = {0:"Collapsed Building", 1:"Fire", 2:"Flood", 3:"Normal"}
    BATCH_SIZE = 8
    IMG_SIZE = (224, 224)
    TRANSFORM_IMG = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMG_SIZE),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225] )
        ])


    preprocess = lambda ims: torch.stack([TRANSFORM_IMG(im.to_pil()) for im in ims])
    explainer = GradCAM(
        model=model,
        target_layer=model.layer4[-1],
        preprocess_function=preprocess
    )



    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    #model = torch.load('../input/pytorch-model/pytorch_res18.pth')
    model.to(device)
    videoCapture = cv.VideoCapture(tfile.name)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    print(fps)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    ps = 25
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoWriter = cv2.VideoWriter(str(i)+"_out.mp4", fourcc, fps, size)
    Q = deque(maxlen=int(fps))
    #with torch.no_grad():
    t1 = time.time()
    c=0
    success, frame = videoCapture.read()
    while success:
        c+=1
        frame_copy = copy.deepcopy(frame) 
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        image_gc = Image(copy.deepcopy(frame_copy))


        # Explain the top label
        explanations = explainer.explain(image_gc)
        #print(explanations)

        image_tensor = TRANSFORM_IMG(frame_copy)

        image_tensor = image_tensor.unsqueeze(0) 
        test_input = image_tensor.to(device)
        outputs = model(test_input)
        _, predicted = torch.max(outputs, 1)
        probability =  F.softmax(outputs, dim=1)
        top_probability, top_class = probability.topk(1, dim=1)
        predicted = predicted.cpu().detach().numpy()
        predicted = predicted.tolist()[0]
        Q.append(predicted)

        results = np.array(Q).mean(axis=0)
        #i = np.argmax(results)
        #print(Q, results, CLASSES[np.round(results)])
        label =CLASSES[np.round(results)]
        top_probability = top_probability.cpu().detach().numpy()
        top_probability = top_probability.tolist()[0][0]
        percentage = top_probability
        top_probability = '%.2f%%' % (top_probability * 100)
        if percentage < 0.45:
            label="Normal"
        heatmap = explanations.explanations[0]['scores']
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img  = cv2.addWeighted(heatmap, 0.3, frame, 0.5, 0)
        if label == "Normal":
            color = (0, 150, 0)
            frame = cv2.putText(frame, label+': '+top_probability, (50, 50), 2, 0.5 ,color, 1 )
            videoWriter.write(frame)
        else:
            color = (0, 0, 150)
            superimposed_img = cv2.putText(superimposed_img, label+': '+top_probability, (50, 50), 2, 0.5 ,color, 1 )
            videoWriter.write(superimposed_img)



        success, frame = videoCapture.read()
    videoWriter.release()

    t2=time.time()
    print('done', t2-t1, c)



