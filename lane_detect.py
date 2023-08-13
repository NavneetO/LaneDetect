#from turtle1 import right
import cv2 
import numpy as np
from matplotlib import pyplot as plt

def regionOfInterest(image):
    height=image.shape[0]
    width=image.shape[1]
    polygons = np.array([[(int(width/6),height),(int(width/2.5),int(height/1.45)),(int(width/1.9),int(height/1.45))]])

    zeroMask = np.zeros_like(image)

    cv2.fillPoly(zeroMask,polygons,1)
    roi=cv2.bitwise_and(image,image,mask=zeroMask)

    return roi
def getLineCoordinate(frame,lines):
    height=int(frame.shape[0]/1.5)
    slope,yIntercept=lines[0],lines[1]

    y1=frame.shape[0]
    y2=int(y1-110)

    x1=int((y1-yIntercept)/slope)
    x2= int((y2-yIntercept)/slope)

    return np.array([x1,y1,x2,y2])

def getLines(frame,lines):
    copyImage=frame.copy()
    leftLine,rightline=[],[]
    roiHeight=int(frame.shape[0]/1.5)
    lineFrame=np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 =line.reshape(4)
            lineData=np.polyfit((int(x1),int(x2)),(int(y1),int(y2)),1)
            slope,yIntercept=lineData[0],lineData[1]
            if slope<0:
                leftLine.append((slope,yIntercept))
            else:
                rightline.append((slope,yIntercept))

    if leftLine:
        leftLineAverage = np.average(leftLine,axis=0)
        left=getLineCoordinate(frame,leftLineAverage)
        #try:
        cv2.line(lineFrame,(left[0],left[1]),(left[2],left[3]),(255,0,0),2)
        #except Exception as e:
        #print('Error',e)
    if rightline:
        rightLineAverage = np.average(rightline,axis=0)
        right=getLineCoordinate(frame,rightLineAverage)
        #try:
        cv2.line(lineFrame,(right[0],right[1]),(right[2],right[3]),(255,0,0),2)
        #except Exception as e:
        #print('Error',e)
    #return np.array([leftLine,rightline])
    return cv2.addWeighted(copyImage,0.8,lineFrame,1,1)

fourcc=cv2.VideoWriter.fourcc('m','p','4','v')
out=cv2.VideoWriter('output.mp4',fourcc,20.0,(640,360))



def new_func(getLines, frame, lines):
    imageWithLine=getLines(frame,lines)
    return imageWithLine
vid=cv2.VideoCapture('E:\\CampK12 VS\\test\\Arrays\\project1\\carDashCam.mp4')
while(vid.isOpened()):
    ret,frame=vid.read()
    cv2.imshow("originalFrame",frame)

    grayFrame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    kernel_size=3
    gaussianFrame=cv2.GaussianBlur(grayFrame,(kernel_size,kernel_size),kernel_size)
    cv2.imshow('gaussianFrame',gaussianFrame)

    low_threshold =75
    high_threshold=100
    edgeframe=cv2.Canny(gaussianFrame,low_threshold,high_threshold)

    roiFrame=regionOfInterest(edgeframe)

    lines=cv2.HoughLinesP(roiFrame,rho=1,theta=np.pi/180,threshold=10,lines=np.array([]),minLineLength=10,maxLineGap=100)
    if lines is not None:
        imageWithLine = new_func(getLines, frame, lines)
    cv2.imshow("final",imageWithLine)

    out.write(imageWithLine)

    if cv2.waitKey(10)& 0xFF==ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()


