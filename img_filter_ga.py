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

def save_list(path_to_file,my_list):
    import io
    # path_to_file = '3-summary_page_scrap_info_limiter/url_summary_missing.txt'
    with io.open(path_to_file, "w",encoding="utf-8") as f:
        for listitem in my_list:
            f.write('%s\n' % listitem)

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
        self.individual_output_img()
        self.fitness_score()
        self.individual_flat_arr2img()
    def individual_output_img(self):
        individual_flat_arr_img = self.w1*self.origin[self.index_w1]+self.w2*self.origin[self.index_w2]+self.w3*self.origin[self.index_w3]
        self.individual_flat_arr_img = individual_flat_arr_img
        return self.individual_flat_arr_img
    def fitness_score(self):
        # self.fitness = self.individual_flat_arr_img -self.target_arr #vector
        # self.fitness = distance.euclidean(ind1.individual_flat_arr_img,ind1.target_arr)
        self.fitness =  np.linalg.norm(self.individual_flat_arr_img -self.target_arr)
        return self.fitness
    def individual_flat_arr2img(self):
        '''
        Returning back to image from flat array and shape
        '''
        output_arr = np.asarray(self.individual_flat_arr_img).reshape(self.shape)
        # output_individual_image = Image.fromarray(output_arr)
        output_individual_image = Image.fromarray((output_arr * 255).astype(np.uint8))
        self.output_individual_image = output_individual_image
        return self.output_individual_image
    def save_ind_output_img(self):
        self.output_individual_image.save('image_dataset/individual_image.png')
    def show_ind_output_img(self):
        self.output_individual_image.show()
    def save_weights_and_indexes(self):
        '''
        Saving the weights for future testing on other images. 
        '''
        weight_val1 = self.w1 * self.index_w1
        weight_val2 = self.w2 * self.index_w2
        weight_val3 = self.w3 * self.index_w3
        weight_list = [weight_val1,weight_val2,weight_val3]
        self.weight_list = weight_list
        path_to_file = ('output/weights/output_weight_list.txt')
        save_list(path_to_file,weight_list)
        output_array = weight_val1 + weight_val2 + weight_val3
        self.weight_arr = output_array
        np.save('output/weights/output_weight_list.npy',output_array)
        return self.weight_list,self.weight_arr



    
#loading Images
original_image = Image.open('image_dataset/canada-best-lakes-moraine-lake.jpg')

population =[]
population_size=100
for _ in range(population_size):
    ind = individual()
    population.append(ind)

#mutation
#weights -> +- 0.0001*k
#index -> binary change. 
class mutation():


#cross-over

#Tests for img class
img = img(original_image)
img.show_original_img()
img.show_gray_scale_img()
img.original
img.original_flat_arr
img.gray_scale
img.gray_scale_flat_arr
img.shape

#tests for individual class
ind1 = individual()
ind1.individual_output_img()
ind1.fitness_score()
ind1.individual_flat_arr2img()
ind1.individual_output_img()
ind1.show_ind_output_img
np.linalg.norm(ind1.individual_flat_arr_img -ind1.target_arr)
from scipy.spatial import distance
distance.euclidean(ind1.individual_flat_arr_img,ind1.target_arr)

print(len(ind1.index_w1))
print(len(ind1.w1))



class IndividualFactory:
    
    def __init__(self, genotype_length: int, fitness_evaluator: FitnessEvaluator):
        self.genotype_length = genotype_length
        self.fitness_evaluator = fitness_evaluator
        # E.g. {:032b} to format a number on 32 bits with leading zeros
        self.binary_string_format = '{:0' + str(self.genotype_length) + 'b}'
    
    def with_random_genotype(self):
        genotype_max_value = 2 ** self.genotype_length
        random_genotype = self.binary_string_format.format(random.randint(0, genotype_max_value))
        fitness = self.fitness_evaluator.evaluate(random_genotype)
        return Individual(random_genotype, fitness)
    
    def with_set_genotype(self, genotype: str):
        fitness = self.fitness_evaluator.evaluate(genotype)
        return Individual(genotype, fitness)
    
    def with_minimal_fitness(self):
        minimal_fitness_genotype = self.binary_string_format.format(0)
        fitness = self.fitness_evaluator.evaluate(minimal_fitness_genotype)
        return Individual(minimal_fitness_genotype, fitness)


#TODO 
'''
-Create the generation
-Create the selection function
-mutation
https://towardsdatascience.com/simple-genetic-algorithm-by-a-simple-developer-in-python-272d58ad3d19
'''





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
