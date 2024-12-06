import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
import tkinter as tk
from tkinter import *
import tkinter.filedialog
from PIL import Image, ImageTk

import math
# ---------------------------- CONSTANTS ------------------------------- #
PINK = "#e2979c"
RED = "#e7305b"
GREEN = "#9bdeac"
YELLOW = "#f7f5dd"
FONT_NAME = "Courier"


# Saves image Canvas pointers
class ImageCanvas:
    def __init__(self):
        self.photoimage = None
        self.canvasImage = None
        self.canvasText = None

    def setImage(self, image):
        self.photoimage = image
    def setCanvasImage(self, image):
        self.canvasImage = image
    def setCanvasText(self, text):
        self.canvasText = text


def is_fruit_ripe_url(img_url, type=0):
    img = cv2.imread(img_url)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    total = np.sum(np.any(img > 0, axis=-1))

    mask = cv2.inRange(hsv, (30, 20, 80), (75, 255, 255))

    masked = np.sum(mask > 0)
    proportion = masked / total
    if proportion < 0.3:
        return "Ripe"
    else:
        return "Unripe"

def is_fruit_ripe(img, type=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV color space
    total = np.sum(np.any(img > 0, axis=-1))

    mask = cv2.inRange(hsv, (30, 20, 80), (75, 255, 255))

    masked = np.sum(mask > 0)
    proportion = masked / total
    if proportion < 0.3:
        return "Ripe"
    else:
        return "Unripe"


def resize_image(img, target_size=(200,200)):
  img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)  # Resize
  return img_resized


def edge_removal_imginput(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_noise_uni = np.random.uniform(0, 255, img.shape)  # Shape manifests

    noise_img = img + 0.01 * img_noise_uni
    gauss_blur = cv2.GaussianBlur(noise_img, (7, 7), 0)
    gauss_blur = cv2.normalize(gauss_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    edges = cv2.Canny(gauss_blur, threshold1=50, threshold2=50)

    kernel = np.ones((4, 4), np.uint8)
    thick_edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(thick_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, color=255, thickness=cv2.FILLED)
    mask_3channel = cv2.merge([mask, mask, mask])

    result = cv2.bitwise_and(img, mask_3channel)
    gained_img_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    retval, gained_img_gray_mask = cv2.threshold(gained_img_gray, 0, 255, cv2.THRESH_BINARY)

    return result, gained_img_gray_mask




def applymethod(image ):
    boolVal = is_fruit_ripe(image)
    text = f"{boolVal}".title()

    # Get the dimensions of the image
    img_height, img_width = image.shape[:2]

    # Calculate the size of the text
    font_scale = 1.5
    font_thickness = 7
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Calculate the position for the text to be centered
    text_x = (img_width - text_size[0]) // 2  # Center horizontally
    text_y = (img_height + text_size[1]) // 2  # Center vertically


    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)



def adjust_gamma(image, gamma=1.0 ):
    invGamma = 1.0 / (gamma+1e-06)
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    given_img = cv2.LUT(image, table)
    applymethod(given_img)
    return given_img

# image uploader function
def imageUploader( ):
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = tk.filedialog.askopenfilename(filetypes=fileTypes)

    if len(path):
        # print(path)
        img = Image.open(path)
        img = img.resize((300, 300))
        pic = ImageTk.PhotoImage(img)
        imageCanvas.setImage(pic)
        canvas.itemconfig(imageCanvas.canvasImage, image=imageCanvas.photoimage)
        canvas.itemconfig(imageCanvas.canvasText, text=f"{is_fruit_ripe_url(path)}")


    else:
        print("Error wrong path.")

def main_algo():
    gamma = 100
    window_name = 'Gamma Correction'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("gamma, (* 0.01)", window_name, gamma, 300, lambda x:x)
    # define a video capture object
    vid = cv2.VideoCapture(1)
    # model = train()
    model = None
    while (True):
        ret, frame = vid.read()
        g = cv2.getTrackbarPos("gamma, (* 0.01)", window_name) * 0.01
        adjusted = adjust_gamma(frame, gamma=g).astype('uint8')
        cv2.imshow(window_name, np.hstack([adjusted]))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()



imageCanvas = ImageCanvas()
window = Tk()
window.title("Are you RIPE or UNRIPE")
window.config(padx=100, pady=50, bg=YELLOW)


title_label = Label(text="DIGIMAP", fg=GREEN, bg=YELLOW, font=(FONT_NAME, 50))
title_label.grid(column=1, row=0)

canvas = Canvas(width=200, height=224, bg=YELLOW, highlightthickness=0)
imageCanvas.setImage( PhotoImage(file="apodous_overripe_1.jpg"))
imageCanvas.setCanvasImage(canvas.create_image(100, 112, image=imageCanvas.photoimage))
imageCanvas.setCanvasText(canvas.create_text(100, 130, text="HI", fill="black", font=(FONT_NAME, 35, "bold")))
canvas.grid(column=1, row=1)

start_button = Button(text="Start Live Video Scan", highlightthickness=0, command=main_algo)
start_button.grid(column=0, row=2)

# defining our upload buttom
uploadButton = tk.Button(window, text="Scan Image Upload", command=imageUploader)
uploadButton.grid(column=2, row=2)




window.mainloop()

