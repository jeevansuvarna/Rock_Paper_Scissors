from keras.models import load_model
import cv2
import numpy as np
from random import choice

REV_CLASS_MAP={
    0:"rock",
    1:"paper",
    2:"scissors",
    3:"none"
}

def mapper(val):
    return REV_CLASS_MAP[val]

def winner(m1,m2):
    if m1==m2:
        return "Tie"
    if m1=="rock":
        if m2=="scissors":
            return "user"
        if m2=="paper":
            return "computer"

    if m1=="paper":
        if m2=="rock":
            return "user"
        if m2=="scissors":
            return  "computer"

    if m1=="scissors":
        if m2=="paper":
            return "user"
        if m2=="rock":
            return "computer"

model=load_model("rps-model.h5")

cap=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)


prev_move =None

while True:
    ret,frame=cap.read()
    if not ret:
        continue

    cv2.rectangle(frame,(100,100),(500,500),(255,255,255),2)
    cv2.rectangle(frame,(800,100),(1200,500),(255,255,255),2)

    roi =frame[100:500,100:500]
    img=cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(227,227))

    pred=model.predict(np.array([img]))
    move=np.argmax(pred[0])
    umovename=mapper(move)

    if prev_move!=umovename:
        if umovename!="none":
            computer_move=choice(['rock','paper','scissors'])
            win=winner(umovename,computer_move)
        else:
            computer_move="none"
            win="waiting......"
    prev_move=umovename


    font =cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,"User's Move:"+umovename,(50,50),font,1.2,(255,255),2,cv2.LINE_AA)
    cv2.putText(frame,"Computer's Move:"+computer_move,(750,50),font,1.2,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame, "Winner: " + win,(400, 600), font, 2, (0, 0, 255), 4, cv2.LINE_AA)

    if computer_move!="none":
        icon=cv2.imread("img/{}.png".format(computer_move))
        icon=cv2.resize(icon,(400,400))
        frame[100:500,800:1200]=icon

    cv2.imshow("ROCK PAPER SCISSORS",frame)

    k=cv2.waitKey(10)
    if k==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()