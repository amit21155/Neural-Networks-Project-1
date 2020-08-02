# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 18:40:45 2018

@author: Amit
"""
import pandas as pd;
import math;
import numpy as np;
import sklearn.preprocessing as skp;
from sklearn.model_selection import train_test_split
import random;

# creating empty list for weights


#step 6 and step 7 : Calculate the number of parameters (weights) you need to tune in the STRUCTURE (refer to slide #2). You need to tune PxN parameters (weights)
def weights(pop_size,Npop):
    weights=[]
    for i in range(0,Npop):
        weights.append(np.random.uniform(low=-1.0, high=1.0, size=pop_size))
    return weights;
        

# calculating the yhat from equation 1
def yhat(weights,x_train_normalize):
    yhat_intermediate=[]
    yhat=[]
    for i in range(0,len(weights)):
        yhat_intermediate.append((np.dot(x_train_normalize,weights[i])))
    for k in range(0,len(yhat_intermediate)):
        sum2=[]
        for i in range(0,len(x_train_normalize)):
        
            sum1=0
            for j in range(0,10):
                sum1=sum1+(1/(1+np.exp(-(yhat_intermediate[k][i][j]))))
            sum2.append(sum1)
        yhat.append(np.array(sum2))
    return yhat;

#yhat for test set    
def yhat_ytest(weights,x_test_normalize):
    yhat=[]
    
    yhat_intermediate = ((np.dot(x_test_normalize,weights)))
    for k in range(0,len(yhat_intermediate)):
        sum1=0
        for l in range(0,10):
            sum1=sum1+(1/(1+np.exp(-(yhat_intermediate[k][l]))))
        yhat.append(round(sum1))
    return yhat;
        
# step 8 : Calculate the fitness_values via Eq(2) for each solution
def fitness_value(yhat,y_train_normalize):    
    fitness=[]
    for j in range(0,len(yhat)):
        sum_fitness=0
        for i in range(0,len(y_train_normalize)):
            sum_fitness=sum_fitness+((yhat[j][i]-y_train_normalize[i])**2)
        fitness.append((1-(sum_fitness/len(y_train_normalize)))*100)
    
    return fitness;


# step 10 :  Binarize all other population according to the following procedure :
def binarize_weight(weights):
    thousand=[]
    rounding=[]
    weights_reshape=[]
    for i in range(0,len(weights)):
        weights_reshape.append(np.array(weights[i]).reshape(1,50))
    w_normalize=[]
    for i in range(0,len(weights_reshape)):
        normalize=[]
        for j in range(0,50):
            normalize.append((weights_reshape[i][0][j]+1)/2)
        w_normalize.append(normalize)
    for i in range(0,len(w_normalize)):
        thousand.append((w_normalize[i]*1000))
    for i in range(0,len(w_normalize)):
        rounding.append(np.round(thousand[i]).astype(int))
    return rounding;

# step 10 : Binarize the parent
def binarize_parent(parent):
    parent_normalizer = skp.MinMaxScaler()
    parent_normalize=(parent_normalizer.fit_transform(parent))
    parent_thousand=((parent_normalize*1000).astype(int))
    parent_round=(np.round(parent_thousand))
    return parent_round;

# step 11 : Making the chromosomes of parent
def chromosome_parent(parent_round):
    parent_reshape=parent_round.reshape(1,50)
    parent_chromosome=''
    for i in range(0,len(parent_reshape[0])):
        parent_chromosome+=str(bin(parent_reshape[0][i])[2:].zfill(10))
    return parent_chromosome;

# step 11 : Making the chromosomes of weights
def weight_chromosome(rounded_weights):
    chromosome_final=[]
    for k in range(0,len(rounded_weights)):
        chromosome=''
        for i in range(0,50):
            chromosome+=str(bin(rounded_weights[k][i])[2:].zfill(10))
        chromosome_final.append(chromosome)
    return chromosome_final;


# step 12 : Crossover function
def crossover (ch1, ch2):
  r = random.randint(2,len(ch1)-1)
  return ch1[:r]+ch2[r:], ch2[:r]+ch1[r:]


# percentage at which the mutation will happen
default_rate=0.05
# step 13 :  Mutation Function
def mutation(ch):
  mutation_list = []
  for i in ch:
    if random.random() < default_rate:
      if i == 1:
        mutation_list.append(0)
      else:
        mutation_list.append(1)
    else:
      mutation_list.append(i)
  return mutation_list;

#step 14 : Debinarization Function
def debinarize_mutationlist(mutation_list,npop):
    desegment_final=[]
    for k in range(0,len(mutation_list)):
        i=0 
        n=10 # no. of elements to be taken
        j=0
        desegment=[]
        while (j<npop):
            desegment.append(mutation_list[k][i:i+n]) # dividing the list into 10 elements
            i=i+10
            j=j+10
        desegment_final.append(desegment)

#Converting the list of chromosomes to string of chromosomes
    string_of_chromosomes_final=[]
    for k in range(0,len(desegment_final)):
        string_of_chromosomes=[]
        for i in range(0,len(desegment_final[0])):
        
            str1=""
            for j in range(0,10):
                str1=str1+str(desegment_final[k][i][j])
            string_of_chromosomes.append(str1)
        string_of_chromosomes_final.append(string_of_chromosomes)

    decimal_final=[]
    for i in range(0,len(string_of_chromosomes_final)):
        decimal=[]
        for j in range(0,len(string_of_chromosomes_final[0])):
            decimal.append( int(string_of_chromosomes_final[i][j], 2))
        decimal_final.append(decimal)
    thousand_div_final=[]
    for i in range(0,len(decimal_final)):
        thousand_div=[]
        for j in range(0,len(decimal_final[0])):
            thousand_div.append((decimal_final[i][j]/1000))
        thousand_div_final.append(thousand_div)
    denormalize_final=[]
    for i in range(0,len(thousand_div_final)):
        denormalize=[]
        for j in range(0,len(thousand_div_final[0])):
            denormalize.append(2*(thousand_div_final[i][j])-1)
        denormalize_final.append(denormalize)
    return denormalize_final;

# Import the dataset from the .csv file provide . 
data=pd.read_csv("Project 1 - Dataset(3).csv");
# Choose N=10 (see STRUCTURE on slide #2) 
N=10;
#Fitness_list to store the fitness values
fitness_list=[];    
#converting to matrix
datamatrix=data.as_matrix();
#converting to array
mydata=np.array(datamatrix)
# choosing the first five columns
input_data=mydata[:,:5]
# choosing the last column as output
output_data=mydata[:,13]
# - Choose 25% of the dataset (random) as testing and the rest 75% as training samples. Leave the testing dataset on the side for the time being
x_train,x_test,y_train,y_test = train_test_split(input_data,output_data,test_size=0.25)
# Normalise the training dataset with values between 0 and 1
min_max_scaler1 = skp.MinMaxScaler()
x_train_normalize = min_max_scaler1.fit_transform(x_train)
y_train_normalize=y_train/np.linalg.norm(y_train, ord=np.inf, axis=0, keepdims=True)
# initial population of parameters
Npop=500
# no. of columns
p=len(x_train_normalize[0])
# population size of weights
pop_size = (p,N)
#calling the weight function to get the weights
weight_values=weights(pop_size,Npop);
# calling the yhat function to get the yhat values
yhat_values=yhat(weight_values,x_train_normalize);
#calling the fitness function to get the fitness values
fitness_values=fitness_value(yhat_values,y_train_normalize);
#taking out the maximum fitness value
fitness_list.append(max(fitness_values));
#taking out the index position  of maximum fitness value 
parent_index=np.argmax(fitness_values);
# taking out the weight of maximum fitness value and making it the parent
parent=weight_values[parent_index];
#binarizing the weights
binary_weight=binarize_weight(weight_values);
#binarizing the parent
binary_parent=binarize_parent(parent);
#making the chromosomes of parent
chromosome_of_parent=chromosome_parent(binary_parent);
#making the chromosomes of weight
chromosome_of_weight=weight_chromosome(binary_weight);
# no. of iterations for step 12 to 17
no_of_iterations=20;
# taking the intitial value of iteration to be 0
itr=0;
#starting the while loop for step 12 to step 17
while(itr<no_of_iterations):
    # creating the offspring empty list to store the offsprings and here the population will become 2*Npop
    offspring=[];
    for j in range(0,len(chromosome_of_weight)):
        off1,off2=crossover(chromosome_of_parent,chromosome_of_weight[j])#calling the crossover function
        offspring.append(off1)#appending first offspring
        offspring.append(off2)#appending second offspring
    mutation_list=[] #creating the empty list to store the mutation results
    for i in range(0,len(offspring)):
        mutation_list.append(mutation(offspring[i]));#calling the mutation function
    debinarize=debinarize_mutationlist(mutation_list,Npop); # debinarizing the mutation list  
    denormalize_weights=[] # empty list to store the denormalized weights
    for i in range(0,len(debinarize)):
        denormalize_weights.append(np.asarray(debinarize[i]).reshape(p,N));# reshaping the weights from (1,p*N) to (p,N)
    storing_index_postition_weight=[] # empty list to store the index position of the weights 
    for i in range(0,len(denormalize_weights)):
        storing_index_postition_weight.append((i,denormalize_weights[i]));# storing the index position of the weights
    yhat_values_new=yhat(denormalize_weights,x_train_normalize); #calculate the yhat values from the denormalized weights
    fitness_values_new=fitness_value(yhat_values_new,y_train_normalize);#calculate the fitness values for the yhat calculated in previous step
    fittest=max(fitness_values_new);# get the maximum fitness value
    parent_index_new=np.argmax(fitness_values_new); # get the index position of maximum fitness value
    parent_new=(denormalize_weights[parent_index_new]); # get the corresponding weight to get the parent
    storing_index_position_parent=[]# list to store the index postion of the fitness values
    for i in range(0,len(fitness_values_new)):
        storing_index_position_parent.append((i,fitness_values_new[i])); # storing the fitness value index postion so we will be having 1000 index postions corresponding to each fitness value 
        
    Fitness_new_sorted=(sorted(storing_index_position_parent, key= lambda x: x[1] ,reverse=True)) # sorting the fitness value list created above on the basis of its value in the reverse direction i.e. descending order 
    Fitness_new_sorted_best=Fitness_new_sorted[1:Npop+1] # taking out the best upper elements and again we have Npop population now
        
    taking_index_position_parent=[] # empty list to store the values of only index position of reduced set of fitness values 
    for i in range(0,len(Fitness_new_sorted_best)):
        taking_index_position_parent.append(Fitness_new_sorted_best[i][0])#storing the index postion of the reduced set of fitness values
    taking_weight_from_index_position_parent=[] # empty list to store the weights according the reduced fitness list index positions
    for i in range(0,len(storing_index_postition_weight)):
        if(storing_index_postition_weight[i][0] in taking_index_position_parent):
            taking_weight_from_index_position_parent.append(storing_index_postition_weight[i][1])# storing the weights according the reduced fitness list index positions and now we have Npop weight matrices
    fitness_list.append(fittest);# appending the maximum fitness value from the current iteration
    binary_weight=binarize_weight(taking_weight_from_index_position_parent);#binarizing the weights
    chromosome_of_weight=weight_chromosome(binary_weight);# making chromosomes of weights
    binary_parent_new=binarize_parent(parent_new);# binarizing the parent
    chromosome_of_parent=chromosome_parent(binary_parent_new);# making the chromosomes of parent
    itr=itr+1;#increasing the value of iteration
    
    
print("Th fitness value list is shown below :")
print(fitness_list)
# Scatter Plot the highest fitness_value for each iteration
import matplotlib.pyplot as plt;
y=[]
for i in range(1,(no_of_iterations+2)):
    y.append(i);
plt.plot(y,fitness_list , 'o', color='black');
plt.xlabel("Fitness_values")
plt.ylabel("iterations")
plt.show;
#Scatter Plot in 3D, the first and second input and the estimated output together
#with real output (y), for testing dataset
min_max_scaler10 = skp.MinMaxScaler()
x_test_normalize = min_max_scaler10.fit_transform(x_test)
y_test_normalize=y_test/np.linalg.norm(y_test, ord=np.inf, axis=0, keepdims=True)
yhat_values_test=yhat_ytest(parent_new,x_test_normalize);
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
fig = pyplot.figure()
ax = Axes3D(fig)
x_vals=x_test[:,0]
y_vals=x_test[:,1]
z1_vals=yhat_values_test
z2_vals=y_test_normalize
ax.scatter(x_vals, y_vals, z1_vals)
ax.scatter(x_vals, y_vals, z2_vals)
pyplot.xlabel("input1")
pyplot.ylabel("input2")
pyplot.show()
# Find out the overall error for testing dataset from below
summation_error=0
for i in range(0, len(y_test)):
    summation_error+=((yhat_values_test[i]-y_test_normalize[i])**2)
error=summation_error/len(y_test)
print("The overall error for testing set is given below :")
print(error)
    
    
        
    
    
    
    
    
    