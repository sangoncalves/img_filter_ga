'''
Load image
Convert into vector

Transform image into grayscale
Convert image to vector

create NN with weights (more info below )

Let the GA choose the weights
 - fitness function: Min distance of Individual to grayscale vector

Objective is create a NN that adds grayscale filter.
'''



from PIL import Image
import numpy as np
import random

import os
path_folder = 'C:/Users/sande/OneDrive/Masters Data Science/4º Semester/Bio-Inspired AI/Project/image_filter/'
os.chdir(path_folder)

def img2flat_arr(image_jpg):
    global shape,flat_arr
    arr = np.array(original_image)
    shape = arr.shape
    flat_arr = arr.ravel()
    return flat_arr


def flat_arr2img(flat_arr):
    global shape
    output_arr = np.asarray(flat_arr).reshape(shape)
    output_image = Image.fromarray(output_arr, 'RGB')
    output_image.save('image_dataset/output_image.png')
    output_image.show()
    return output_image





#Getting the vector for the images
# original_image_arr = img2flat_arr(original_image)
# target_image_arr = img2flat_arr(target_image)

# img.save('image/greyscale.png')

class img():
    def __init__(self, image):
        global shape,original_flat_arr,gray_scale_flat_arr
        #original
        self.original = image
        original_flat_arr = img2flat_arr(image) #initial image
        self.original_flat_arr = original_flat_arr
        self.shape = shape
        #gray scale
        self.gray_scale = image.convert('LA')
        gray_scale_flat_arr = img2flat_arr( self.gray_scale) #target
        self.gray_scale_flat_arr = gray_scale_flat_arr
    def save_original_img(self):
        self.original.save('image_dataset/original_img.png')
    def show_original_img(self):
        self.original.show()
    #GRAY SCALE 
    def save_gray_scale_img(self):
        self.gray_scale.save('image_dataset/gray_scale_img.png')
    def show_gray_scale_img(self):
        self.gray_scale.show()

class individual():
    def __init__(self):
        global shape,original_flat_arr,gray_scale_flat_arr
        self.shape = shape
        self.origin = original_flat_arr #origin = original from class img
        self.target_arr = gray_scale_flat_arr # target = gray_scale from class img
        self.len_flat_arr = len(self.origin)
        # self.w1, self.w2, self.w3=np.arange(0,1,0.0001),np.arange(0,1,0.0001),np.arange(0,1,0.0001)
        self.w1, self.w2, self.w3=random.uniform(0, 1),random.uniform(0, 1),random.uniform(0, 1)
        self.index_w1 = np.random.randint(2, size=len(self.origin))
        self.index_w2 = np.random.randint(2, size=len(self.origin))
        self.index_w3 = np.random.randint(2, size=len(self.origin))
    def individual_output_img(self):
        individual_flat_arr_img = self.w1*self.origin[self.index_w1]+self.w2*self.origin[self.index_w2]+self.w3*self.origin[self.index_w3]
        self.individual_flat_arr_img = individual_flat_arr_img
        return self.individual_flat_arr_img
    def fitness_score(self):
        # self.fitness = self.individual_flat_arr_img -self.target_arr
        self.fitness =  np.linalg.norm(self.individual_flat_arr_img -self.target_arr)
        return self.fitness
    def individual_flat_arr2img(self): #TODO work on it!
        '''
        Returning back to image from flat array and shape
        '''
        output_arr = np.asarray(self.individual_flat_arr_img).reshape(self.shape)
        output_individual_image = Image.fromarray(output_arr)
        self.output_individual_image = output_individual_image
        return self.output_individual_image
    def save_gray_scale_img(self):
        self.output_individual_image.save('image_dataset/individual_image.png')
    def show_gray_scale_img(self):
        self.output_individual_image.show()
    



#loading Images
original_image = Image.open('image_dataset/canada-best-lakes-moraine-lake.jpg')
# target_image = original_image.convert('LA')

# original_image.show()

#Testing
img = img(original_image)
img.show_original_img()
img.show_gray_scale_img()
img.original
img.original_flat_arr
img.gray_scale
img.gray_scale_flat_arr
img.shape

ind1 = individual()
ind1.individual_output_img()
ind1.fitness_score()
ind1.individual_flat_arr2img()
ind1.show_gray_scale_img()

ind1.individual_output_img()
ind1.show_gray_scale_img


np.linalg.norm(ind1.individual_flat_arr_img -ind1.target_arr)
distance.euclidean(ind1.individual_flat_arr_img,ind1.target_arr)

len(ind1.index_w1)
len(ind1.w1)
# ind1 = individual()

# test = img_individual(original_image)
# test.flat_arr2img()
# test.img_show()
# target.flat_arr2img()
# target = img(target_image)



#Create weights
'''
#original_image * weight = gray_scale_image

#This weights probably need in genetic form? 

# we could let the "network" decide what to multiply

w1* [random indexis of the original image] + w2* [random indexis of the original image] + w3* [random indexis of the original image]. Also, W can assume random value from 0 to 1. 

The objective is the NN learn that the value it needs to assume is Y = 0.2125 R + 0.7154 G + 0.0721 B
https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

'''
#Create fitness function
    #Min dist(individual, gray_scale_imag)

#Create individuals
    #mutation+ crossover

#Create selection mechanism

#iterate
