#!/usr/bin/env python
# coding: utf-8

# In[1]:


from imageai.Classification import ImageClassification
import os

execution_path = os.getcwd()

prediction = ImageClassification()
prediction.setModelTypeAsInceptionV3()
prediction.setModelPath("inception_v3_google-1a9a5a14.pth")
prediction.loadModel()

predictions, probabilities = prediction.classifyImage("dog2.jpeg", result_count=10)
for eachPrediction, eachProbability in zip(predictions, probabilities):
    print(eachPrediction , " : " , eachProbability)

