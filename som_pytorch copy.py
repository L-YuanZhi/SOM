import torch
import os
import math

from cv2 import cv2
 
import matplotlib.pyplot as plt 
import numpy as np 

class self_organizing_map:
    
    def __init__(self, m,n,dims,dataset,max_epoch=None,initial_radius=None,initial_learning_rate=None,one_pixel_rate=None):
        """
        initialize the som model
        """
        self.__m = m
        self.__n = n
        self.__dims = dims
        if max_epoch!=None and max_epoch>0 and type(max_epoch)==int:
            self.__max_epoch = max_epoch
            self.__epoch = 0
        else:
            raise AttributeError("max_epoch is required, max_epoch must be int type and bigger th an 0")
        self.__somap = None
        self.__weights = None
        self.__global_step = None
        self.__divide_param = None
        self.__label_map = None #NEW

        if one_pixel_rate == None: # contral the effect arae is only one pixel, with % in hole training process
            self.__one_pixel_rate = 0.9
        elif one_pixel_rate>1.0:
            raise AttributeError("parameter one_pixel_rate must smaller than 1.0")
            self.__one_pixel_rate = one_pixel_rate  
        
        self.__dataset = dataset # tensor[]
        self.__initial_var = 1.0
        if initial_radius == None: 
            self.__initial_radius = 0.5*min(m,n)
        else:
            self.__initial_radius = initial_radius

        if initial_learning_rate==None:
            self.__initial_learning_rate =  0.1
        else:
            self.__initial_learning_rate = initial_learning_rate
        self.__Create_map() 
        # print("MAP",self.__somap)

    def __Create_map(self):
        """
        create a location list of map with shape (m*n,2) 
        and create a list of weights with shape (m*n,dims),
        and initial elemnets from 0 to 1
        """
        # weights list with shape of (m*n,dims)
        self.__weights = torch.rand(self.__m*self.__n,self.__dims)
        # location list of weights with shape (m*n,2)
        self.__somap = []
        for x in range(self.__m):
            for y in range(self.__n):
                self.__somap.append((x,y))
        np.array(self.__somap)
        self.__global_step = 0

    def Train(self):
        """
        training process of the som model
        """
        if self.__dataset == None:
            raise AttributeError("dataset is empty, dataset must be created before train")
        
        self.__One_pixel()

        for i in range(self.__max_epoch):
            # shuffle and load the training data
            data_loader = torch.utils.data.DataLoader(self.__dataset,shuffle=True)
            self.__epoch = i+1    
            # print("Epoches:",self.__epoch,"/",self.__max_epoch)
            #finding best matching units
            #load one image from dataset
            for element in data_loader:
                self.__global_step+=1
                # print(":",self.__global_step)
                #BMU ==> ((x,y),index)
                
                bmu_location = self.__Bmu(element)
                #list of effect area ==> [((x,y),distance),...]
                #calculate the effect area of map
                effect_area = self.__Effect_area(bmu_location)
                #update weights in the area
                self.__Weights_update(effect_area,element)
            #looping until epoch reach the given max value

    def __Bmu(self,input_tensor):
        """
        find the best match unit of input tensor in form as ((x,y),index)\n
        :param input_tensor: input tensor\n
        :returns: location of the best matching units\n
        """
        distances = [] 
        # calculation of the distances 
        for element in self.__weights:
            # inper-product
            # distances.append(float(torch.sum(element*input_tensor)))
            # euclidean distance
            distances.append(self.__Eu_distance(input_tensor,element))
        distances = np.array(distances)
        # print("distances",distances)
        # find the index of bmu
        # bmu_list = np.argwhere(distances==np.min(distances))
        # bmu_list = bmu_list.reshape(1,)
        # print("BMU location",self.__somap[np.argmin(distances)])
        return self.__somap[np.argmin(distances)]

    def __Eu_distance(self,vect1,vect2):
        """
        find the best match unit of input tensor
        :param vect1: input tensor, 1-dimension train vector
        :param vect2: input tensor, 1-dimension weight vector and must be the same length of vect1 
        :returns: euclidean distance of inputs 
        """
        return math.sqrt(torch.sum((vect1-vect2)**2))


    # def __Bmu_locations(self,bmu_index):
    #     """
    #     return a list of 2-dimension locations of the best matching units
    #     :param bmu_index: list include bmu indices of ceertain vector.
    #     """
    #     return self.__somap[bmu_index]
    
    def __Weights_update(self,effect_area,input_tensor):
        """
        update all weights inside the effect area of bmu
        """
        # m(t+1) = m(t)+h(t)[x(t)-m(t)]
        for (x,y), distance in effect_area:
            #calculate learning rate for each weights
            learning_rate = self.__Learning_rate(distance)
            #weights value update
            self.__weights[self.__m*x+y]+=learning_rate*(input_tensor[0]-self.__weights[self.__m*x+y])
            # print("weights",self.__weights[self.__m*x+y])
    
    def __Effect_area(self,bmu_location):
        """
        return the effected area, witch is centered at the bmu
        :param bmu_location: location in map of the best matching units
        :return location_list: return a list of the effect area, 
                               and a list of distance between elements and the center
                               ((x,y),distance)
        """
        raidus = self.__initial_radius*math.exp(-1.*self.__global_step/self.__divide_param)
        locations = []
        cx,cy = bmu_location
        for (x,y) in self.__somap:
            if (x-cx)**2+(y-cy)**2<=raidus**2:
                locations.append(((x,y),(x-cx)**2+(y-cy)**2))
        # print("locations",locations) 
        
        return locations
        
    def __Learning_rate(self,distance):
        """
        return the learning rate of this epoch
        :param distance: distance to the center of effect area
        """
        #var decrease with time
        # gaussian rate of effect area
        var = self.__initial_var*math.exp(-1.*self.__global_step/self.__divide_param)
        if var < 0.001:
            var = 0.001
        const = 1/(self.__initial_var*(math.sqrt(2*math.pi)))
        gaussian_rate=math.exp(-(distance**2)/(2*var**2))
        gaussian_rate=const*gaussian_rate

        # global_rate=initial_rate*math.exp(-self.__global_step/self.__divide_param)
        global_rate = self.__initial_learning_rate*(1-(self.__global_step/(self.__max_epoch*len(self.__dataset))))
        if global_rate < 0.0001:
            global_rate = 0.0001
        # print("RATE",var,const, gaussian_rate,global_rate,gaussian_rate*global_rate)

        return gaussian_rate*global_rate

    def Weights_output(self):
        """
        return the weights of self organizing map
        """
        return self.__weights

    def Test(self, input_tensor):
        return self.__Bmu(input_tensor)
    
    def __One_pixel(self):
        """
        make the training process effect only the BMU in setup %
        """
        max_step = len(self.__dataset)*self.__max_epoch
        certain_point = -1.*self.__one_pixel_rate*max_step
        self.__divide_param = certain_point/math.log(0.5/self.__initial_radius)

    def Weights_Input(self,input_tensor):
        self.__weights = input_tensor