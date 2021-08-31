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
def mutate(individual):
    print('NOT MUTATED: ',len(individual.index_w1),len(individual.index_w2),len(individual.index_w3))
    population = []
    prob1 = random.uniform(0, 1)
    if prob1>0.6:
        if  individual.w1<=0.9995:
            # individual.mutated_w1 = individual.w1[index]+0.0005
            w1_mutated = individual.w1+0.0005
        else:
            # individual.mutated_w1 = individual.w1[index]-0.0005
            w1_mutated = individual.w1-0.0005
    else:
        w1_mutated = individual.w1
    prob2 = random.uniform(0, 1)
    if prob2>0.6:
        if  individual.w2<=0.9995:
            # individual.mutated_w1 = individual.w1[index]+0.0005
            w2_mutated = individual.w2+0.0005
        else:
            # individual.mutated_w1 = individual.w1[index]-0.0005
            w2_mutated = individual.w2-0.0005
    else:
        w2_mutated = individual.w2
    prob3 = random.uniform(0, 1)
    if prob3>0.6:
        if  individual.w3<=0.9995:
            # individual.mutated_w1 = individual.w1[index]+0.0005
            w3_mutated = individual.w3+0.0005
        else:
            # individual.mutated_w1 = individual.w1[index]-0.0005
            w3_mutated = individual.w3-0.0005
    else:
        w3_mutated = individual.w3
    iw1_list = []
    for index_weight in individual.index_w1:
        prob = random.uniform(0, 1)
        if prob>0.6:
            if index_weight==0:
                iw1_list.append(1)
            else:
                iw1_list.append(0)
    iw2_list = []
    for index_weight in individual.index_w2:
        prob = random.uniform(0, 1)
        if prob>0.6:
            if index_weight==0:
                iw2_list.append(1)
            else:
                iw2_list.append(0)
    iw3_list = []
    for index_weight in individual.index_w3:
        prob = random.uniform(0, 1)
        if prob>0.6:
            if index_weight==0:
                iw3_list.append(1)
            else:
                iw3_list.append(0)
    # print(w1_mutated,w2_mutated,w3_mutated,iw1_list,iw2_list,iw3_list)
    print('MUTATED: ',len(iw1_list),len(iw2_list),len(iw3_list))
    new_ind = new_individual(w1=w1_mutated,w2=w2_mutated,w3=w3_mutated,index_w1=iw1_list,index_w2=iw2_list,index_w3=iw3_list)
    population.append(new_ind)
    return population      





class new_individual():
    def __init__(self,w1,w2,w3,index_w1,index_w2,index_w3):
        global shape,original_flat_arr,gray_scale_flat_arr
        self.shape = shape
        self.origin = original_flat_arr #origin = original from class img
        # print('original_flat_arr: ',original_flat_arr)
        self.target_arr = gray_scale_flat_arr # target = gray_scale from class img
        self.len_flat_arr = len(self.origin)
        self.w1, self.w2, self.w3=w1,w2,w3
        self.index_w1,self.index_w2,self.index_w3 = index_w1,index_w2,index_w3
        print('new_individual: ',len(index_w1),len(index_w2),len(index_w3))
        # print( index_w1,index_w2,index_w3)
        self.individual_output_img()
        self.fitness_score()
        self.individual_flat_arr2img()
    def individual_output_img(self):
        individual_flat_arr_img = self.w1*self.origin[self.index_w1]+self.w2*self.origin[self.index_w2]+self.w3*self.origin[self.index_w3]
        # print('individual_flat_arr_img: ',individual_flat_arr_img)
        self.individual_flat_arr_img = individual_flat_arr_img
        return self.individual_flat_arr_img
    def fitness_score(self):
        # self.fitness = self.individual_flat_arr_img -self.target_arr #vector
        # self.fitness = distance.euclidean(ind1.individual_flat_arr_img,ind1.target_arr)
        self.fitness =  np.linalg.norm(self.individual_flat_arr_img -self.target_arr)
        print(self.fitness)
        return self.fitness
    def individual_flat_arr2img(self):
        '''
        Returning back to image from flat array and shape
        '''
        print('shape: ',shape)
        output_arr = np.asarray(self.individual_flat_arr_img).reshape(self.shape)
        # print('output_arr: ',output_arr)
        # output_individual_image = Image.fromarray(output_arr)
        output_individual_image = Image.fromarray((output_arr * 255).astype(np.uint8))
        # print('output_individual_image: ', output_individual_image)
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

img = img(original_image)
ind_test = individual()
# np.linalg.norm(ind_test.individual_flat_arr_img -ind_test.target_arr)
w1,w2,w3 = ind_test.w1, ind_test.w2, ind_test.w3
index_w1,index_w2,index_w3 = ind_test.index_w1,ind_test.index_w2,ind_test.index_w3
n_ind_set_param = new_individual(w1,w2,w3,index_w1,index_w2,index_w3)
new_population = mutate(n_ind_set_param)



n_ind_set_param.index_w2
len(n_ind_set_param.index_w2)
for ind in n_ind_set_param.index_w2:
    print(ind)
    break
new_population = mutate(ind_test)

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
