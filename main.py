import numpy as np
import math
import matplotlib.pyplot as plt
import copy

# comment delete me

PRECISION = 1e-2
NAME = "massif5.txt"

class element(object):
    def __init__(self, name, x, y):
        self.x = x
        self.y = y
        self.name = name
    def inf(self):
        print("\n  "+self.name+"\n("+str(self.x)+", "+str(self.y)+")\n")

#   Read elements from text file
def read_elements(name):
    f = open(name)
    text = f.readlines()
    v_size = len(text)
    line = []
    el = []
    for i in range(v_size):
        line.append(text[i])
        line[i] = line[i].replace('\n', '')
        line[i] = line[i].split(sep = ' ')
        for j in range(len(line[i])):
            el.append(element(line[i][j],j,-i))
    return el

#   Fill zerous v*h
def fill_zeros(v,h):
    el_zero = [ [element("Z0", 0, 0)]*h for i in range(v)]
    return el_zero

#   Displays the position of all elements
def el_print(el):
    y = []
    x = []
    for i in range(len(el)):
        y.append(el[i].y)
        x.append(el[i].x)
    el_new = fill_zeros(-min(y)+1,max(x)+1)
    for i in range(-min(y)+1):
        el_temp = []
        el_temp2 = []
        for j in range(len(el)):
            if el[j].y == -i:
                el_temp.append(el[j])
        for j in range(len(el_temp)):
            if el_temp[j].x == j:
                el_temp2.append(el_temp[j])
        el_new[i] = el_temp2
    
    for i in range(len(el_new)):
        text = ""
        for j in range(len(el_new[i])):
            text = text + el_new[i][j].name + " "
        text = text + "\n"
        print(text)

#   Rturn array element names
def el_list(el):
    el_names = []
    for i in range(len(el)):
        flag = 0
        for j in range(len(el_names)):
            if el_names[j] == el[i].name:
                flag = 1
        if flag == 0:
            el_names.append(el[i].name)
    el_names.sort()
    el_name_count = np.zeros(len(el_names))
    for i in range(len(el)):
        for j in range(len(el_names)):
            if el_names[j]==el[i].name:
                el_name_count[j] = el_name_count[j] + 1
    return el_names

#   Rturn array element counts
def el_count(el):
    el_name=el_list(el)
    el_name_count = np.zeros(len(el_name))
    for i in range(len(el)):
        for j in range(len(el_name)):
            if el_name[j]==el[i].name:
                el_name_count[j] = el_name_count[j] + 1
    return el_name_count

#   Find metrics for current fi in this position
def metrics_for_fi(el, fi):
    x, y = [], []
    for i in range(len(el)):
        y.append(el[i].y)
        x.append(el[i].x)
    C = np.array([max(x)+1, min(y)-1])/2
    R = np.linalg.norm(C)
    A = np.array([C[0]-R*math.cos(fi),C[1]-R*math.sin(fi)])
    a, b = A[0]-C[0], A[1]-C[1]
    c = C[0]**2 + C[1]**2 - R**2 - A[0]*C[0]-A[1]*C[1]
    el_name= el_list(el)
    metrics = np.zeros(len(el_name))
    for i in range(len(el)):
        for j in range(len(el_name)):
            if el_name[j]==el[i].name:
                metrics[j] = metrics[j]+abs(a*(el[i].x+0.5)+b*(el[i].y-0.5)+c)/(R*math.sqrt(a**2+b**2))    
    el_name_count = el_count(el)
    metrics= metrics/el_name_count
    return metrics

#   Find metrics for current elements position
def get_metrics(el):
    pi = math.pi
    el_name = el_list(el)
    r = [ []*1 for i in range(len(el_name))]
    dr = [ []*1 for i in range(len(el_name))]
    FI = np.arange(0, 2*pi, pi*PRECISION)
    for fi in FI:
        for i in range(len(el_name)):
            r[i].append(metrics_for_fi(el, fi)[i])
            dr[i].append(abs(metrics_for_fi(el, fi)[i]-1)*pi*PRECISION)
    for i in range(len(el_name)):
        dr[i] = sum(dr[i])
    
    el_name_count = el_count(el)
    Metrics = sum(dr*el_name_count)/sum(el_name_count)
    return Metrics

#   Plot diagram
def plot_graph(el):
    pi = math.pi
    el_name = el_list(el)
    r = [ []*1 for i in range(len(el_name))]
    dr = [ []*1 for i in range(len(el_name))]
    FI = np.arange(0, 2*pi, pi*PRECISION)
    for fi in FI:
        for i in range(len(el_name)):
            r[i].append(metrics_for_fi(el, fi)[i])
            dr[i].append(abs(metrics_for_fi(el, fi)[i]-1)*pi*PRECISION)
    for i in range(len(el_name)):
        dr[i] = sum(dr[i])
        
    ax = plt.subplot( projection='polar')
    ax.set_rlabel_position(0)
    ax.plot(FI, np.ones(len(FI)), '--', linewidth=2,color='green',alpha=.5, label = 'ideal')
    ax.fill_between(FI, np.ones(len(FI)),color='g',alpha=.05)
    for i in range(len(el_name)):
        ax.plot(FI, r[i], linewidth=4,alpha=.6, label = el_name[i]+" "+str(round(dr[i],1)))
    ax.grid(True)
    ax.legend(loc='upper left')
    plt.show()

#   Fill filled_massif.txt the current element
def fill_text(el):
    y = []
    x = []
    for i in range(len(el)):
        y.append(el[i].y)
        x.append(el[i].x)
    el_new = fill_zeros(-min(y)+1,max(x)+1)
    for i in range(-min(y)+1):
        el_temp = []
        el_temp2 = []
        for j in range(len(el)):
            if el[j].y == -i:
                el_temp.append(el[j])
        for j in range(len(el_temp)):
            if el_temp[j].x == j:
                el_temp2.append(el_temp[j])
        el_new[i] = el_temp2
    
    text = ""
    for i in range(len(el_new)):
        temp = ""
        for j in range(len(el_new[i])):
            temp = temp + el_new[i][j].name + " "
        temp = temp + "\n"
        text = text+temp
    f = open('filled_massif.txt','w')
    f.write(text)
    f.close()

def swap_el(el,i,j):
    temp_name = el[i].name
    el[i].name   = el[j].name 
    el[j].name   = temp_name
    return el

def permutations(arr, index=0):
    global min_metric
    global best_position
    if index != len(arr):
        used = set()
        for i in range(index, len(arr)):
            if arr[i].name not in used:
                used.add(arr[i].name)
                arr[index].name, arr[i].name = arr[i].name, arr[index].name
                permutations(arr, index + 1)
                arr[index].name, arr[i].name = arr[i].name, arr[index].name
    # else:
    #     metric = get_metrics(arr)
    #     if (min_metric>metric):
    #         min_metric = metric
    #         best_position = copy.deepcopy(arr)

#---------------------------------
####      Main program        ####
#---------------------------------

el = read_elements(NAME)

min_metric = get_metrics(el)

print(min_metric)
el_print(el)
print("**")

permutations(el)

print(min_metric)
# el_print(best_position)
print("**")

# def permutations(arr, index=0):
#     global min_metric
#     global best_position
#     if index != len(arr):
#         used = set()
#         for i in range(index, len(arr)):
#             if arr[i].name not in used:
#                 used.add(arr[i].name)
#                 arr[index].name, arr[i].name = arr[i].name, arr[index].name
#                 permutations(arr, index + 1)
#                 arr[index].name, arr[i].name = arr[i].name, arr[index].name
#     else:
#         metric = get_metrics(arr)
#         if (min_metric>metric):
#             min_metric = metric
#             best_position = copy.deepcopy(arr)

# -------------------- чисто перебор без вычислений --------------------
# def permutations(arr, index=0):
#     global min_metric
#     global best_position
#     if index != len(arr):
#         used = set()
#         for i in range(index, len(arr)):
#             if arr[i].name not in used:
#                 used.add(arr[i].name)
#                 arr[index].name, arr[i].name = arr[i].name, arr[index].name
#                 permutations(arr, index + 1)
#                 arr[index].name, arr[i].name = arr[i].name, arr[index].name
#     # else:
#     #     metric = get_metrics(arr)
#     #     if (min_metric>metric):
#     #         min_metric = metric
#     #         best_position = copy.deepcopy(arr)
