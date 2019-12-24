import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import random
import numpy as np

model = load_model("model.h5")

model.summary()

def load_image(img_path):
    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)         
    img_tensor /= 255.                                      
    return img_tensor

classes = ["Parasitized","Uninfected"]
choice = random.randint(0,1)
path = os.path.join(os.getcwd(),"cell_images/test/" + classes[choice])
images = os.listdir(path)
image_path = os.path.join(path,images[random.randint(0,len(images)-1)])

print(image_path)
print(model.predict(load_image(image_path)))