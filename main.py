import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import os
from tracker import*

model=YOLO('best.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
     

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
            
        




cap=cv2.VideoCapture('headcount.avi')


my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker=Tracker()
up={}
down={}
counterup=[]
counterdown=[]
cy1=180
cy2=245
offset=8
while True:    
    ret,frame = cap.read()
    if not ret:
       break
   

   
    frame=cv2.resize(frame,(1020,500))
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
   

#    print(px)
    
    list=[]    
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        
        d=int(row[5])
        c=class_list[d]
        list.append([x1,y1,x2,y2])
#        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    bbox_idx=tracker.update(list)
    for id,rect in bbox_idx.items():
        x3,y3,x4,y4=rect
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        if cy2<(cy+offset) and cy2>(cy-offset):
           up[id]=cx,cy
        if id in up:
           if cy1<(cy+offset) and cy1>(cy-offset): 
              cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
              cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
              cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
              if counterup.count(id)==0:
                 counterup.append(id)
################################################DOWN##############################
        if cy1<(cy+offset) and cy1>(cy-offset):
           down[id]=cx,cy
        if id in down:
           if cy2<(cy+offset) and cy2>(cy-offset): 
              cv2.circle(frame,(cx,cy),4,(255,0,0),-1)
              cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
              cvzone.putTextRect(frame,f'{id}',(x3,y3),1,1)
              if counterdown.count(id)==0:
                 counterdown.append(id)
    cv2.line(frame,(1,180),(1017,180),(255,0,255),1)

    cv2.line(frame,(1,245),(1017,245),(0,0,255),1)
         


    countingup=(len(counterup))
    countingdown=(len(counterdown))
    cvzone.putTextRect(frame,f'{countingup}',(50,60),1,1)
    cvzone.putTextRect(frame,f'{countingdown}',(50,100),1,1)

    
    
   

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()    
cv2.destroyAllWindows()




