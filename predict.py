from tensorflow.keras.models import load_model
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

classes = os.listdir('./train')
model = load_model("ftmodel.h5")

def predict(input_filename):

    input = image.load_img(input_filename,target_size=(224,224))

    input = np.expand_dims(input,axis=0)

    input = preprocess_input(input)

    result = model.predict(input)

    label = classes[np.argmax(result[0])]

    probability = result[0][np.argmax(result[0])]*100

    return label, probability

if __name__ == "__main__":
    test_filename = "./test"
    test_classes = os.listdir(test_filename)
    count=0
    acc_count=0
    for i in test_classes :
        for j in os.listdir(test_filename+"/"+i):
            count+=1
            x, y = predict(test_filename+"/"+i+"/"+j)
            if x == i:
                acc_count+=1

    print(acc_count/count)
