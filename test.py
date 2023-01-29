# Import required Libraries
from tkinter import *
from PIL import Image, ImageTk
import cv2
import torch
from matplotlib import pyplot as plt
import numpy as np
import easyocr
from tqdm import tqdm

# Create an instance of TKinter Window or frame
win= Tk()

cap= cv2.VideoCapture('test/2.jpg')
#cap= cv2.VideoCapture(0)

# Graphics window
imageFrame = Frame(win, width=500, height=500)
imageFrame.grid(row=0, column=0, padx=10, pady=2)

processFrame = Frame(win, width=500, height=500)
processFrame.grid(row=0, column=500, padx=10, pady=2)

croppedFrame = Frame(win, width=500, height=100)
croppedFrame.grid(row=500,column=500, padx=10, pady=2, sticky='w')


# Set the size of the window
lblOriginal = Label(imageFrame, text="Orignal", font=('Arial', 18))
lblOriginal.grid(row=0,column=0)
label =Label(imageFrame)
label.grid(row=1, column=0)

lblProcessing = Label(processFrame, text="Processing", font=('Arial', 18))
lblProcessing.grid(row=0, column=0)
label2 = Label(processFrame)
label2.grid(row=1, column=0)

lblOutput = Label(croppedFrame, text='Detected Number', font=('Arial', 18))
lblOutput.grid(row=0, column=0)

lblCropped = Label(croppedFrame)
lblCropped.grid(row=1, column=0)

lblNumber = Label(croppedFrame, text='Number', font=('Arial', 18))
lblNumber.grid(row=1, column=50)

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/last.pt', force_reload=True)
global results
# Define function to show frame
def show_frames():
    
    # Processing
    cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
    #cv2image = cv2.resize(cv2image, (500,500), interpolation = cv2.INTER_AREA)
    grey = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
    
    img = Image.fromarray(cv2image)
    
    results = model(cv2image)        # <====================================
    
    processImg = np.squeeze(results.render())
    
    img2 = Image.fromarray(processImg)
    img2 = img2.resize((500,500))

    # Convert image to PhotoImage
    img = img.resize((500,500))
    imgtk = ImageTk.PhotoImage(image = img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    #label.after(20, show_frames)
    
    imgtk2 = ImageTk.PhotoImage(image = img2)
    label2.imgtk = imgtk2
    label2.configure(image=imgtk2)
    
    df = results.pandas().xyxy[0]
    df = df.drop(['class', 'name','confidence'], axis=1)
    values = df.values.astype(int)
    values = np.squeeze(values)
    
    croppedImg = cv2image[values[1]:values[3], values[0]:values[2]]
    
    img3 = Image.fromarray(croppedImg)
    imgtk3 = ImageTk.PhotoImage(image = img3)
    lblCropped.imgtk = imgtk3
    lblCropped.configure(image=imgtk3)
    
    reader = easyocr.Reader(['en'])
    num = reader.readtext(croppedImg)
    print(num)
    for x in num:
        detectNumber = x[1].strip()
        lblNumber['text'] = detectNumber
    
show_frames()

win.mainloop()
cap.release()
