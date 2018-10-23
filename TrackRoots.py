import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
import numpy as np
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from scipy.ndimage import label, generate_binary_structure
from tkinter import Tk
from pylab import imshow
from tkinter.filedialog import askopenfilename
from tkinter import *
from PIL import Image, ImageTk
import pandas as pd
imgFiles = []

def TrackRoots(RootCords, img, file1):
    import MainFunction2 as mf #moved import to allow for cyclical imports in python. see -> https://gist.github.com/datagrok/40bf84d5870c41a77dc6

    initXCords = RootCords[0]
    initYCords = RootCords[1]
    initCords = RootCords
    pathPrefix = file1[:-6]

    print(RootCords, initXCords, initYCords)
    print(img)

    #loop through each coordinate and trace along the root (analyzing image for bright areas),
    #  measuring distance between prev and current tip @ each frame
    global imgFiles
    stages = []
    for x in range(49):
        stages.append("Point"+str(x+1))
        path = pathPrefix + "t" + str(x+1) + ".TIF"
        imgFiles.append(path)
    print("FILE PATHS:", "\n", imgFiles)

    roots = []
    rtStages = []

    for k in range(len(RootCords[0])):
        current = [RootCords[0][k], RootCords[1][k]]
        roots.append(current)

    rtStages.append(roots)

    print(rtStages)

    stage2 = []
    for each in roots:
        stage2.append(traceRoot(each))

    rtStages.append(stage2)

    #############################
    master2 = Tk()
    #Convert the images into a JPG for use in Tkinter window
    image = Image.open(mf.toJPG(imgFiles[1], 0))
    
    #Set up structure for window
    windowStructure = mpimg.imread(imgFiles[1])
    if len(windowStructure.shape) < 3:
        windowStructure = mf.to_rgb1a(windowStructure)
        
    width1,height1,dim = windowStructure.shape  #Create an image object to initialize the size of the canvas window
    
    cImage = image.convert("RGBA")

    w = Canvas(master2, width=height1, height=width1) #intialize the canvas on the external window with a size of the image being displayed
    w.pack()
    #Plot the image

    imgT = ImageTk.PhotoImage(cImage)
    w.create_image(0,0,anchor="nw",image=imgT)

    master2.mainloop()

    ##############################
    
    dataf = {'R1x': "", 'R1y': "", 'R2x': "", 'R2y': "" , 'R3x': "",'R3y': "", 'R4x': "",'R4y': "", 'R5x': "",'R5y': "", 'R6x': "",'R6y': "",'R7x': "",'R7y': "",'R8x': "",'R8y': "",'R9x': "",'R9y': "",'R10x': "",'R10y': ""}
    df = pd.DataFrame(dataf, index=stages)

    df.to_csv("./Test2.csv")


def traceRoot(rootTip):
    
    global imgFiles
    stage2f = imgFiles[1]

    print(stage2f)

    return [None, None] #IN PROGRESS - Should return the corddinates of the next root tip after stage1


