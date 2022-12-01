

#import matplotlib.pyplot as plt
!pip install tensorflow

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


model = torch.load('pytorch_res18_200.pth')
#model.eval()


# In[ ]:


import torch
from torchvision import transforms
import torchvision.models as models
import cv2
import torch.nn.functional as F
import copy


f = st.file_uploader("Upload file")
tfile = tempfile.NamedTemporaryFile(delete=False)
tfile.write(f.read())



CLASSES = {0:"collapsed_building", 1:"fire", 2:"flood", 3:"normal"}
BATCH_SIZE = 8
IMG_SIZE = (224, 224)
TRANSFORM_IMG = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(IMG_SIZE),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

# model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = torch.load('../input/pytorch-model/pytorch_res18.pth')
model.to(device)
#model.load_state_dict(torch.load('checkpoint.pt'))
#model.eval()

Q = deque(maxlen=8)

#for i in os.listdir('../input/test-video2/'):
#    print(i)
videoCapture = cv.VideoCapture(tfile.name)

#videoCapture = cv2.VideoCapture('../input/test-video2/'+i)
fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
ps = 25
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
videoWriter = cv2.VideoWriter(str(i)+"_out.mp4", fourcc, fps, size)

with torch.no_grad():
    success, frame = videoCapture.read()
    while success:
        frame_copy = copy.deepcopy(frame) 
        frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
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
        top_probability = '%.2f%%' % (top_probability * 100)

        frame = cv2.putText(frame, label+': '+top_probability, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        videoWriter.write(frame)
        success, frame = videoCapture.read()
    videoWriter.release()
print('done')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




