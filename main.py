from utils import *
import cv2
import time



w,h=640,480
pid=[0.5,0.5,0]
pError=0
myDrone = initTello()
startCounter=0  ## 1 = no flight
flag=0

cpt=1
while True:
    ##Takeoff
    if startCounter==0:
        myDrone.takeoff()
        time.sleep(8)
        myDrone.move_up(80)
        startCounter=1
    ##Step1
    frame = telloGetframe(myDrone,w,h)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            cpt=cpt+1
    if cpt%2==0:
        frame, info = findface(frame)
        pError = trackFace(myDrone, info, w, h, pid, pError)
    else:
        Sudoku(frame,flag)
    ##Step3


    #print(area)
    #Flight(myDrone)


    cv2.imshow('IMG',frame)
    if cv2.waitKey(1) & 0xFF == ord('e'):
         myDrone.land()
         break