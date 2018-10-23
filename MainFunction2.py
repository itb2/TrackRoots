
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
from TrackRoots import TrackRoots


#Plot the RGB image and save the plotted image as a JPG file which can be read into Tk.OpenImage Function
def toJPG(filePath, which):   #Read in filepath
    print(filePath)
    cutfile = filePath[:-3]   #Filename without the picture type(.bmp/.tif/jpg etc)
    fileJPG = (cutfile + "jpg")   #Add the JPG file type to original image path
    img = mpimg.imread(filePath)      #Convert Filepath to image
    print(len(img.shape))

    fig = plt.figure()              #Plot the RGB image and save the plotted image as a JPG file
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False) # this removes the ticks and numbers for x axis
    ax.get_yaxis().set_visible(False) # this removes the ticks and numbers for y axis
    fig.add_axes(ax)

    if len(img.shape) == 2:         #Create new variable holding the potential new RGB image
        imgRGB = to_rgb1a(img)
    else:
        imgRGB = img

    if which == 1:                  #Set 1st image color to luminous heat map...
        lum_img = imgRGB[:,:,0]
        plt.imshow(lum_img, cmap="Blues", interpolation="none")
    elif which == 2: 
        lum_img = imgRGB[:,:,0]
        plt.imshow(lum_img, cmap="hot", interpolation="none")
    else:
        plt.imshow(imgRGB, interpolation="none")


    plt.savefig(fileJPG,bbox_inches="tight", dpi=110)
    plt.close()
    
    return fileJPG
    
   
def to_rgb1a(im):
    # This should be fsater than 1, as we only
    # truncate to uint8 once (?)
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] =  ret[:, :, 1] =  ret[:, :, 0] =  im
    return ret

master = Tk()
#Set up initial image variables
imagePath1 = askopenfilename()
imagePath2 = askopenfilename()

#Convert the images into a JPG for use in Tkinter window
image1 = Image.open(toJPG(imagePath1, 1))
image2 = Image.open(toJPG(imagePath2, 2))

#Set up structure for window
windowStructure = mpimg.imread(imagePath1)
if len(windowStructure.shape) < 3:
    windowStructure = to_rgb1a(windowStructure)
    
width1,height1,dim = windowStructure.shape  #Create an image object to initialize the size of the canvas window


#Create the external windoww


#Convert images from RGB to RGBA 
PREimg = image1.convert("RGBA")
PREimg2 = image2.convert("RGBA")
print(PREimg.size)
print(PREimg2.size)

#Merge the two images
c = Image.blend(PREimg, PREimg2, 0.4)


#Counter for mouseclick events
count = 1
#Arrays to hold coordinate values and points
listOfXC = []
listOfYC = []
listOfP = []


w = Canvas(master, width=height1, height=width1) #intialize the canvas on the external window with a size of the image being displayed
w.pack()
 #Plot the image

img = ImageTk.PhotoImage(c)
w.create_image(0,0,anchor="nw",image=img)
#w.create_image(20,20,anchor="nw",image=img2)
#Create a function that plots a point and coordinates on the image
def clickAndPrint(event):
        #outputting x and y coords to console
        global count
        global listOfXC
        global listOfYC     
        global listOfP
        pos = ("Point " + str(count))
        listOfP.append(pos)
        xcoor = event.x
        ycoor = event.y
        listOfXC.append(xcoor)
        listOfYC.append(ycoor)
        print (pos, "coordinates are: ", str(xcoor),",",str(ycoor))
        w.create_oval(event.x-3,event.y-3, event.x+3, event.y+3, fill="yellow")
        if(count > -1):
            w.create_text(event.x+20,event.y-5,text=("Point " + str(count)))
            count +=1

    #mouseclick event
w.bind("<Button 1>",clickAndPrint)

master.mainloop()


dataf = {'X Coordinates':listOfXC,'Y Coordinates':listOfYC}
df = pd.DataFrame(dataf, index=listOfP)

print(df)

#df.to_csv("Test.csv")

coordinates = [listOfXC, listOfYC];
#print(TrackRoots)
TrackRoots(coordinates, img, imagePath1);

