from keras.models import load_model
import cv2
import numpy as np
import sys

file=sys.argv[1]

REV_CLASS_MAP={
    0:"rock",
    1:"paper",
    2:"scissors",
    3:"none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

model = load_model("rps-model.h5")

img =cv2.imread(file)
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img = cv2.resize(img,(227,227))

predict=model.predict(np.array([img]))
move_code=np.argmax(predict[0])
move_name =mapper(move_code)

print("predicted:{}".format(move_name))