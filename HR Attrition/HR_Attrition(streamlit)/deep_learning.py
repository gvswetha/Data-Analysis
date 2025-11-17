import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
import numpy as np
import matplotlib.pyplot as plt

from simple_image_download import simple_image_download as simp

response = simp.simple_image_download

keywords = ["monkey animal", "bear animal"]
for kw in keywords:
    response().download(kw, 200)
    