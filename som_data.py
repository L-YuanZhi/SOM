import os 
import torch 
import math 
import cv2 as cv2
import numpy as np 
import matplotlib.pyplot as plt 
import data_normalizer_c as dnc

class som_data:
    def __init__(self, rehsape_rate = 1.0):
        self.__dataset = None
        self.__set_len = None
        self.__image_size = None
        self.__data_dims = None
        self.__reshape_rate = rehsape_rate
    
    def Parameter(self):
        """
        output number and dimension of data\n
        and return as tuple (dims,num)\n 
        """
        print("number of data:",self.__set_len)
        print("dimension of data:",self.__data_dims)
        print("image size:",self.__image_size)
        return (self.__data_dims,self.__set_len,self.__image_size)

    def Dataset(self, data_path, mode=None, sta=5,sto=355,ste=5):
        """
        create an dataset for the som
        :param data_path: the directory path of the pipe images
        :param mode: normalize the image with average to 0 and variance to 1 as default with keyword "zero-one",
                     keyword "min-max" nomalize image pixel range from 0 to 1  
        :returns: an dataset contain data agumrnted images, each data will be same shape as ([flatten intensity],...)
        """
        dataset = []
        
        for dirs in os.listdir(data_path):

            if dirs == "bp":
                label = 0
            else:
                label = 1

            for fileName in os.listdir(os.path.join(data_path,dirs)):
                input_image = plt.imread(os.path.join(data_path,dirs,fileName)) # original image for training the model
                #
                size = round(self.__reshape_rate*input_image.shape[0])
                image = cv2.resize(input_image,(size,size))
                #
                self.__image_size = image.shape
                if mode == None or mode == "zero-one":
                    normalized_image = self.__Normalize_circle(image) # normalized image
                elif mode == "min-max":
                    normalized_image = self.__Normalize_circle_minmax(image)
                if sto*ste != 0:
                    for angle in range(sta,sto,ste): # augmenting train data by rotating and fliping the normalized image
                        rotatedImage = self.__rotation(normalized_image,angle)
                        dataset.append(np.append(rotatedImage.flatten(),label))
                        dataset.append(np.append(cv2.flip(rotatedImage,1).flatten(),label))
                else:
                    dataset.append(np.append(normalized_image.flatten(),label))
                    
        # dataset = tuple(dataset)
        self.__set_len=len(dataset)
        self.__data_dims=len(dataset[0])

        return torch.tensor(dataset)

    def Dataset_S(self,data_path,mode=None,sta=5,sto=355,ste=5):
        """
        create an dataset for the som, base on one class
        :param data_path: the directory path of the pipe images
        :param mode: normalize the image with average to 0 and variance to 1 as default with keyword "zero-one",
                     keyword "min-max" nomalize image pixel range from 0 to 1  
        :returns: an dataset contain data agumrnted images, each data will be same shape as ([flatten intensity],...)
        """
        dataset = []
        
        if data_path[:-2] == "bp":
            label = 0
        else:
            label = 1

        for fileName in os.listdir(data_path):
            input_image = plt.imread(os.path.join(data_path,fileName)) # original image for training the model
            size = round(self.__reshape_rate*input_image.shape[0])
            image = cv2.resize(input_image,(size,size))
            self.__image_size = image.shape
            if mode == None or mode == "zero-one":
                normalized_image = self.__Normalize_circle(image) # normalized image
            elif mode == "min-max":
                normalized_image = self.__Normalize_circle_minmax(image)
            if sto*ste != 0:
                for angle in range(sta,sto,ste): # augmenting train data by rotating and fliping the normalized image
                    rotatedImage = self.__rotation(normalized_image,angle)
                    dataset.append(np.append(rotatedImage.flatten(),label))
                    dataset.append(np.append(cv2.flip(rotatedImage,1).flatten(),label))
            else:
                dataset.append(np.append(normalized_image.flatten(),label))

        # dataset = tuple(dataset)
        self.__set_len=len(dataset)
        self.__data_dims=len(dataset[0])

        return torch.tensor(dataset)

    def Dataset_PN(self,data_path,mode=None):
        """
        create an dataset for the som
        :param data_path: the directory path of the pipe images
        :param mode: normalize the image with average to 0 and variance to 1 as default with keyword "zero-one",
                     keyword "min-max" nomalize image pixel range from 0 to 1  
        :returns: an dataset contain data agumrnted images, each data will be same shape as ([flatten intensity],...)
        """
        dataset = []
        dataset_b = []
        dataset_g = []
        for dirs in os.listdir(data_path):
            if dirs == "bp":
                label = 0
            else:
                label = 1

            for fileName in os.listdir(os.path.join(data_path,dirs)):
                input_image = plt.imread(os.path.join(data_path,dirs,fileName)) # original image for training the model
                #
                size = round(self.__reshape_rate*input_image.shape[0])
                image = cv2.resize(input_image,(size,size))
                #
                # image = self.__position_norm(image)
                self.__image_size = image.shape
                if mode == None:
                    normalized_image = image
                elif mode == "zero-one":
                    normalized_image = self.__Normalize_circle(image) # normalized image
                elif mode == "min-max":
                    normalized_image = self.__Normalize_circle_minmax(image)

                pn_image = self.__position_norm(normalized_image)
                dataset.append(np.append(pn_image.flatten(),label))
                if label == 0:
                    dataset_b.append(np.append(pn_image.flatten(),label))
                else:
                    dataset_g.append(np.append(pn_image.flatten(),label))
                    
        # dataset = tuple(dataset)
        self.__set_len=len(dataset)
        self.__data_dims=len(dataset[0])

        return torch.tensor(dataset),torch.tensor(dataset_b),torch.tensor(dataset_g) 
        

    def Dataset_mtm(self,data_path,value=85):

        dataset = []

        for dirs in os.listdir(data_path):
            if dirs == "bp":
                label = 0
            else:
                label = 1

            for fileName in os.listdir(os.path.join(data_path,dirs)):
                input_image = plt.imread(os.path.join(data_path,dirs,fileName)) # original image for training the model
                #
                size = round(self.__reshape_rate*input_image.shape[0])
                image = cv2.resize(input_image,(size,size))
                #
                # image = self.__position_norm(image)
                self.__image_size = image.shape
                
                pn_image = self.__position_norm(image)

                mtm_image = dnc.move_average(pn_image,value)
                
                dataset.append(np.append(mtm_image.flatten(),label))
                    
        # dataset = tuple(dataset)
        self.__set_len=len(dataset)
        self.__data_dims=len(dataset[0])

        return torch.tensor(dataset) 


    def __Normalize_circle(self, input, t_ave=0, t_var=1):
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

    def __Normalize_circle_minmax(self,image,start=0.,stop=1.):
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

    
    def __rotation(self,image,angle):
        """
        rotate the input image with certain angle
        Arg:
            image: the input gray scale image, with np.array type
            angle: the degree which is the image need to be rotate
        Returns:
            rotate: the result image
        """
        size = image.shape[0]
        half_size = size/2
        matRotate = cv2.getRotationMatrix2D((half_size,half_size),angle,1)
        rotate = cv2.warpAffine(image,matRotate,(size,size),0,0,0)
        # for x,y in np.argwhere(rotate==0):
        #     if(x-half_size)**2+(y-half_size)**2>(half_size-10)**2:
        #         rotate[x,y] = random.randint(7,12)
        return rotate

    def __position_norm(self,image):
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

        return self.__rotation(image,angle)
