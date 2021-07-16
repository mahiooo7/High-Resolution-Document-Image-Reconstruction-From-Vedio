from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
import numpy as np
from tkinter import filedialog
import json
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim



main = tkinter.Tk()
main.title("High-Quality Document Image Reconstruction from Video")
main.geometry("1300x1200")

global filename
global ref_id, video_x_len, video_y_len, image_x_len, image_y_len, top_right_y, top_right_x
global bottom_left_y, bottom_left_x, bottom_right_y, bottom_right_x, top_left_y, top_left_x

def upload():
    global filename
    filename = filedialog.askdirectory(initialdir = "smartdoc17-dataset-sample")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+' Selected dataset loaded\n')


def readReferences():
    global ref_id,video_x_len,video_y_len,image_x_len,image_y_len,top_right_y,top_right_x
    global bottom_left_y,bottom_left_x,bottom_right_y,bottom_right_x,top_left_y,top_left_x
    text.delete('1.0', END)
    f = open(filename+'/task_data.json')
    data = json.load(f)
    ref_id = data.get('reference_frame_id')
    text.insert(END,"Image Reference ID : "+str(ref_id)+"\n")
    if ref_id == 0:
        ref_id = 1
    dic = data.get('input_video_shape')
    video_x_len = dic.get('x_len')
    video_y_len = dic.get('y_len')
    text.insert(END,"Video Shape X : "+str(video_x_len)+" Y : "+str(video_y_len)+"\n")

    dic = data.get('target_image_shape')
    image_x_len = dic.get('x_len')
    image_y_len = dic.get('y_len')
    text.insert(END,"Image Shape X : "+str(image_x_len)+" Y : "+str(image_y_len)+"\n")

    dic = data.get('object_coord_in_ref_frame')
    dic1 = dic.get('top_right')
    top_right_y = dic1.get('y')
    top_right_x = dic1.get('x')
    text.insert(END,"Top Right X : "+str(top_right_x)+" Y : "+str(top_right_y)+"\n")            

    dic1 = dic.get('bottom_left')
    bottom_left_y = dic1.get('y')
    bottom_left_x = dic1.get('x')
    text.insert(END,"Bottom Left X : "+str(bottom_left_x)+" Y : "+str(bottom_left_y)+"\n")  

    dic1 = dic.get('bottom_right')
    bottom_right_y = dic1.get('y')
    bottom_right_x = dic1.get('x')
    text.insert(END,"Bottom Right X : "+str(bottom_right_x)+" Y : "+str(bottom_right_y)+"\n")  

    dic1 = dic.get('top_left')
    top_left_y = dic1.get('y')
    top_left_x = dic1.get('x')
    text.insert(END,"Top Left X : "+str(top_left_x)+" Y : "+str(top_left_y)+"\n")  
         
def orderWarpedPoints(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def HomographicPerspectiveTransform(image, pts):
    rect = orderWarpedPoints(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def computeSimilarity(imageA, imageB):
    return ssim(cv2.cvtColor(imageA,cv2.COLOR_BGR2GRAY),cv2.cvtColor(imageB,cv2.COLOR_BGR2GRAY))              
    
def extractDocument():
    pts = [(top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (bottom_left_x, bottom_left_y)]
    pts = np.array(pts, dtype = "float32")
    original = cv2.imread(filename+'/reference_frame_01_dewarped.png')
    similar = 0
    high = None
    refs = None
    videofile = filename+'/input.mp4'
    video = cv2.VideoCapture(videofile)
    while(True):
        ret, frame = video.read()
        if ret == True:
            frame_no = video.get(cv2.CAP_PROP_POS_FRAMES)
            dst = HomographicPerspectiveTransform(frame,pts)
            dst = cv2.rotate(dst,  cv2.ROTATE_90_CLOCKWISE)
            original = cv2.resize(original,(dst.shape[1],dst.shape[0]))
            score = computeSimilarity(original,dst)
            if score > similar:
                similar = score
                high = dst
            print(str(frame_no)+" high score : "+str(similar)+" current frame score"+str(score))
            if frame_no == ref_id:
                refs = frame
                cv2.imwrite("reference_image.png",frame)
                cv2.imwrite("high_resoultion_image.png",high)
            cv2.imshow("video", frame)
            if cv2.waitKey(50) & 0xFF == ord('q'):
                break    
        else:
            break
    video.release()
    cv2.destroyAllWindows()
    cv2.imshow("Ground Truth image",original)
    cv2.imshow("Reference image",refs)
    cv2.imshow("High Resolution Document Image",high)
    cv2.waitKey(0)
    


def close():
    main.destroy()
    
font = ('times', 16, 'bold')
title = Label(main, text='High-Quality Document Image Reconstruction from Video')
title.config(bg='PaleGreen2', fg='Khaki4')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Sample Videos", command=upload)
upload.place(x=700,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=700,y=150)

noiseButton = Button(main, text="Read File Details", command=readReferences)
noiseButton.place(x=700,y=200)
noiseButton.config(font=font1) 

cnnButton = Button(main, text="Run & Extract High Quality Image", command=extractDocument)
cnnButton.place(x=700,y=250)
cnnButton.config(font=font1) 

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=700,y=300)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)


main.config(bg='PeachPuff2')
main.mainloop()
