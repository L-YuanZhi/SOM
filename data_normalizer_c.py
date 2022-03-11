import numpy as np
import cv2 as cv2
import math


def Normalize_circle(input, t_ave=0, t_var=1):
    """
    normalize the image but only with area inside the pipe circle 
    by move the average to the similar value, default average 0 and variance 1

    :param input: the input image of the pipe
    :param t_ave: target average 
    :param t_var: target variance

    :returns: an normalized image
    """
    output = np.zeros(input.shape[:2],np.float)
    pixels = []
    average = 0

    if input.shape[0]!=input.shape[1]:
        raise AttributeError("The width and height of the pipe image must be the same")
    else:
        size = input.shape[0]
        center = round(size/2)
        radius = int(center-10)

    for w,h in np.argwhere(input>=0):
        if (w-center)**2+(h-center)**2<=radius**2:
            pixels.append((w,h))
            average += input[w,h]
    average = average/len(pixels)

    variance = 0
    for w,h in pixels:
        variance += (input[w,h]-average)**2
    variance = math.sqrt(variance/len(pixels))
    
    for w,h in pixels:
        output[w,h] = t_ave + t_var*(input[w,h]-average)/variance
    
    return output

def Normalize_circle_minmax(image,start=0.,stop=1.):
    """
    normalize the image but only with area inside the pipe circle 
    by move minimum value to start and maximum to stop

    :param image: the input image of the pipe
    :param start: the minimum value of image, default as 0.
    :param stop: the maximum value of image, default as 1.

    :returns: an normalized image
    """
    if start<0 or stop>255:
        raise Warning("Result may not be save as required form")

    if image.shape[0]!=image.shape[1]:
        raise AttributeError("The width and height of the pipe image must be the same")
    else:
        size = image.shape[0]
        center = round(size/2)
        radius = int(center-10)

    output = np.zeros(image.shape[:2],np.float32)

    pmin = 255
    pmax = 0

    for w,h in np.argwhere(output==0):
        if (w-center)**2+(h-center**2)<=radius**2:
            if image[w,h]<pmin:
                pmin = image[w,h]
            if image[w,h]>pmax:
                pmax = image[w,h]
    
    for w,h in np.argwhere(output==0):
        if (w-center)**2+(h-center**2)<=radius**2:
            output[w,h] = ((image[w,h]-pmin)/(pmax-pmin))*(stop-start)+start
    
    return output

def position_norm(image):
    """
    make the brightest pixel position on the 90 digree line
    Arg:
        image: the input image, np.array type
    """
    cx,cy = image.shape
    cx=int(cx/2)
    cy=int(cy/2)
    bright_p = np.unravel_index(np.argmax(image,axis=None),image.shape)

    dx = bright_p[0]-cx
    dy = bright_p[1]-cy
    if dx > 0:
        angle = 180 + math.atan(-dy/dx)*180/math.pi
    else:
        angle = math.atan(-dy/dx)*180/math.pi
    # print(angle)

    return rotation(image,angle)

def rotation(image,angle):

    cx,cy = image.shape
    cx=int(cx/2)
    cy=int(cy/2)
    
    matRotate = cv2.getRotationMatrix2D((cx,cy),angle,1)
    rotate = cv2.warpAffine(image,matRotate,image.shape,0,0,0)

    return rotate

def move_average(image,value):
    mean = np.mean(image)
    return image-mean+value

# if __name__ = "__main__":
