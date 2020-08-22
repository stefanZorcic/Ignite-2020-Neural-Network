# Recurrent Neural Network
# Goal to have sentimet recognizition

#Modlues
import numpy as np
import random
import matplotlib.pyplot as plt
  
def sigmoid(x):
  return 1.0/(1.0+np.exp(x))

#Chain rule = dError/dw = dError/douto * douto/dino * dino/dw

def derivative1(outo, targetOutput):
  return outo-targetOutput

def derivative2(outo):
  return outo*(1-outo)
  

