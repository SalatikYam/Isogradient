import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from random import shuffle
import random

PRECISION = 1e-2
NAME = "massif4.txt"

# read structure in txt file
def read_strucrure(name):
    f = open(name)
    text = f.readlines()
    v_size = len(text)
    line = []
    char_matrix = []
    int_matrix = []
    for i in range(v_size):
        line.append(text[i])
        line[i] = line[i].replace('\n', '')
        line[i] = line[i].replace('A', '')
        line[i] = line[i].split(sep = ' ')
        char_matrix.append(line[i])
    for i in char_matrix:
        int_matrix.append([int(j) for j in i])
    matrix = np.zeros([len(int_matrix),len(int_matrix[0])], int)-1
    # делает заполнение np массива
    for i in range(len(int_matrix)):
        for j in range(len(int_matrix[i])):
            matrix[i][j] = int_matrix[i][j]
    
    return matrix + 1

# give numbers array matrix and their counts
def get_num_and_count(matrix):
    numbers = set(np.ravel(matrix))
    if 0 in numbers:
        numbers.remove(0)
    numbers = list(numbers)
    numbers = np.array(numbers)
    counts = np.zeros(len(numbers), int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(numbers.shape[0]):
                if(matrix[i][j]==numbers[k]):
                    counts[k]= counts[k] + 1
    return numbers, counts

# give metric, vectors, abs_vectors
def get_metric(matrix):
    center = np.array([matrix.shape[1]/2,-matrix.shape[0]/2])
    numbers, counts = get_num_and_count(matrix)
    vectors = np.zeros([len(numbers),2])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            for k in range(numbers.shape[0]):
                if(matrix[i][j]==numbers[k]):
                    vectors[k]= vectors[k] + np.array([j + 0.5,-i - 0.5])-center
    vectors = np.transpose(np.transpose(vectors)*counts)
    abs_vectors = np.array([[(vectors[i][0]**2+vectors[i][1]**2)**(1/2)] for i in range(len(vectors))])
    # print('***')
    # print(np.transpose(abs_vectors))
    # print(np.transpose(counts))
    # print(np.dot(np.transpose(abs_vectors),counts))
    # print('***')

    # print('***')
    # # print(vectors)
    # # print(counts)
    # # print(np.transpose(np.transpose(vectors)*counts))
    # # print(vectors*np.transpose(counts))
    # print('***')

    metric = np.dot(np.transpose(abs_vectors),counts)
    return metric, vectors, abs_vectors

# find sector where vector
def find_sector(vector):
    if (-vector[0]<=0 and -vector[1]>=0):
        sector = 1
    elif (-vector[0]>=0 and -vector[1]>=0):
        sector = 2
    elif (-vector[0]<=0 and -vector[1]<=0):
        sector = 3
    else:
        sector = 4
    return sector

# make swep with borders
def sweep(matrix, i1,j1, i2,j2):
    vert, hor = matrix.shape 
    # if (i2>=0 and j2>=0 and i2<vert and j2<hor and matrix[i2][j2]!=0):
    if (i2>=0 and j2>=0 and i2<vert and j2<hor):
        temp = matrix[i1][j1]
        matrix[i1][j1] = matrix[i2][j2]
        matrix[i2][j2] = temp
    return matrix

# return best metric and best matrix (in h area near martix[i][j])
def chek_move(matrix, i, j, sector, h):
    matrix_temp = np.copy(matrix)
    best_matrix = np.copy(matrix)
    best_metric,_ , _ = get_metric(matrix_temp)
    for m in range(h+1):
        for n in range(h+1):
            if   sector == 1:
                matrix_temp = sweep(matrix_temp, i,j, i-m,j-n)
                metric,_ , _ = get_metric(matrix_temp)
                if metric<best_metric:
                    best_matrix = np.copy(matrix_temp)
                    best_metric = metric
                # print('m =', m, 'n=',n, 'metric=',metric)
                # print(matrix_temp)
                matrix_temp = np.copy(matrix)
            elif sector == 2:
                matrix_temp = sweep(matrix_temp, i,j, i-m,j+n)
                metric,_ , _ = get_metric(matrix_temp)
                if metric<best_metric:
                    best_matrix = np.copy(matrix_temp)
                    best_metric = metric
                # print('m =', m, 'n=',n, 'metric=',metric)
                # print(matrix_temp)
                matrix_temp = np.copy(matrix)
            elif sector == 3:
                matrix_temp = sweep(matrix_temp, i,j, i+m,j-n)
                metric,_ , _ = get_metric(matrix_temp)
                if metric<best_metric:
                    best_matrix = np.copy(matrix_temp)
                    best_metric = metric
                # print('m =', m, 'n=',n, 'metric=',metric)
                # print(matrix_temp)
                matrix_temp = np.copy(matrix)
            else:
                matrix_temp = sweep(matrix_temp, i,j, i+m,j+n)
                metric,_ , _ = get_metric(matrix_temp)
                if metric<best_metric:
                    best_matrix = np.copy(matrix_temp)
                    best_metric = metric
                # print('m =', m, 'n=',n, 'metric=',metric)
                # print(matrix_temp)
                matrix_temp = np.copy(matrix)
    # print('best_metric=', best_metric)
    # print(best_matrix)
    return best_matrix, best_metric

# move swep in h area and decrease metric
def gradient_descent(matrix,h=1):
    # print(matrix)
    # h = 5
    numbers, _ = get_num_and_count(matrix)
    matrix_copy = np.copy(matrix)
    best_matrix = np.copy(matrix_copy)
    best_metric, vectors, abs_vectors = get_metric(matrix)
    # находит индекс с наибольшим значением
    max_indices = np.where(abs_vectors == np.amax(abs_vectors))
    # print(max_indices[0][0])
    sector = find_sector(vectors[max_indices[0][0]])
    # print(vectors)
    # print(sector)
    # print(best_metric)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j]==numbers[max_indices[0][0]]:
                matrix_copy, metric = chek_move(matrix, i, j, sector, h)
                if metric < best_metric:
                    best_matrix = np.copy(matrix_copy)
                    best_metric = metric
    # print(max_indices[0][0])
    
    # print(matrix)
    # print(best_matrix)
    # print(best_metric)

    return best_matrix, best_metric

def shuf(matrix):
    len = np.ravel(matrix)
    random.shuffle(len)
    matrix = len.reshape(matrix.shape[0], matrix.shape[1])
    return matrix

#---------------------------------
####      Main program        ####
#---------------------------------

matrix=read_strucrure(NAME)

print(matrix)
print('***')#ff
# matrix=shuf(matrix)
# print(matrix)

# best_matrix, best_metric = gradient_descent(matrix,4)
# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix,4)
#     # print(best_metric)
# print(best_metric)
# print(best_matrix)

matrix=shuf(matrix)
best_matrix, best_metric = gradient_descent(matrix,6)
for i in range(500):
    best_matrix, best_metric = gradient_descent(best_matrix,6)
    # print(best_metric)
print(best_metric)
print(best_matrix)

matrix=shuf(matrix)
best_matrix, best_metric = gradient_descent(matrix,6)
for i in range(500):
    best_matrix, best_metric = gradient_descent(best_matrix,6)
    # print(best_metric)
print(best_metric)
print(best_matrix)







# -------------------- Test --------------------
# used = set(np.ravel(matrix))
# print(used)

# center = np.array([matrix.shape[1]/2,-matrix.shape[0]/2])
# print(center)

# print("vectors:\n",vectors)
# print("abs_vectors:\n",abs_vectors)
# print("metric:\n", metric)

# print("sector:\n",find_sector(vectors[1]))

# print(sweep(matrix, 1,1, 2,5))

# metric, vectors, abs_vectors = get_metric(matrix)

# metric, _, _ = get_metric(matrix)
# print(metric)

# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix)
#     print(best_metric)

# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix)
#     print(best_metric)

# print("vectors:\n",vectors)
# print("abs_vectors:\n",abs_vectors)
# print("metric:\n", metric)

# best_matrix, best_metric = gradient_descent(matrix)
# print(best_metric)
# best_matrix, best_metric = gradient_descent(best_matrix)
# print(best_metric)
# best_matrix, best_metric = gradient_descent(best_matrix)
# print(best_metric)
# best_matrix, best_metric = gradient_descent(best_matrix)
# print(best_metric)
# best_matrix, best_metric = gradient_descent(best_matrix)
# print(best_metric)
# print('*****')
# best_matrix, best_metric = gradient_descent(best_matrix)
# # best_matrix, best_metric = gradient_descent(best_matrix)
# # print(best_metric)

# print(get_num_and_count(matrix))
# metric, vectors, abs_vectors = get_metric(matrix)
# print(metric)
# print(vectors)
# print(find_sector(vectors[1]))
# print(chek_move(matrix, 1, 1, 3, 2))
# gradient_descent(matrix)

# best_matrix, best_metric = gradient_descent(matrix)
# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix)
#     # print(best_metric)
# print(best_matrix)
# metric, vectors, abs_vectors = get_metric(best_matrix)
# print(vectors)

# matrix = shuffle(matrix)
# new = np.ravel(matrix)
# print(new)
# random.shuffle(new)
# print(new)
# random.shuffle(matrix)
# print(matrix)
# print('***')

# best_matrix, best_metric = gradient_descent(matrix)
# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix)
#     # print(best_metric)
# print(best_metric)
# print(best_matrix)

# matrix=shuf(matrix)
# best_matrix, best_metric = gradient_descent(matrix,4)
# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix,4)
#     # print(best_metric)
# print(best_metric)
# print(best_matrix)

# matrix=shuf(matrix)
# best_matrix, best_metric = gradient_descent(matrix,4)
# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix,4)
#     # print(best_metric)
# print(best_metric)
# print(best_matrix)

# matrix=shuf(matrix)
# best_matrix, best_metric = gradient_descent(matrix,4)
# for i in range(100):
#     best_matrix, best_metric = gradient_descent(best_matrix,4)
#     # print(best_metric)
# print(best_metric)
# print(best_matrix)

