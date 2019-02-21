"""
ECE196 Face Recognition Project
Author: Will Chen

Prerequisite: You need to install OpenCV before running this code
The code here is an example of what you can write to print out 'Hello World!'
Now modify this code to process a local image and do the following:
1. Read geisel.jpg
2. Convert color to gray scale
3. Resize to half of its original dimensions
4. Draw a box at the center the image with size 100x100
5. Save image with the name, "geisel-bw-rectangle.jpg" to the local directory
All the above steps should be in one function called process_image()
"""

import cv2

# TODO: Edit this function
def process_image():
    img = cv2.imread('geisel.jpg',0) #zero is in greyscale
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    oX = int(img.shape[0]/2)
    oY = int(img.shape[1]/2)
    
    img = cv2.rectangle(img,(oY+50,oX-50),(oY-50,oX+50),(255,0,0),4) 

    cv2.imwrite("challenge2.jpg",img) 

    return img

# Just prints 'Hello World! to screen.
def hello_world():
    print('Hello World!')
    return

# TODO: Call process_image function.
def main():
    process_image()
    return 0



if(__name__ == '__main__'):
    main()
