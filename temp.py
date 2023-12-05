import numpy as np
import math
import matplotlib.pyplot as plt
PRECISION = 1e-2


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





def plot_1(y1,y2,y3,y4):
    #ax.set_rlabel_position(0)
    ax = plt.subplot()
    plt.xlabel("градусы")
    plt.ylabel("Tset[псек]")
    ax.plot(x, y1, '-*', linewidth=2,alpha=.5, label = 'Ne')
    ax.plot(x, y2, '-*', linewidth=2,alpha=.5, label = 'Ar')
    ax.plot(x, y3, '-*', linewidth=2,alpha=.5, label = 'Kr')
    ax.plot(x, y4, '-*', linewidth=2,alpha=.5, label = 'Xe')

    ax.grid(True)
    ax.legend(loc='upper left')
    plt.show()

def plot_2(y5,y6,y7,y8):
    #ax.set_rlabel_position(0)
    ax = plt.subplot()
    plt.xlabel("градусы")
    plt.ylabel("фКл")
    ax.plot(x, y5, '-*', linewidth=2,alpha=.5, label = 'Ne')
    ax.plot(x, y6, '-*', linewidth=2,alpha=.5, label = 'Ar')
    ax.plot(x, y7, '-*', linewidth=2,alpha=.5, label = 'Kr')
    ax.plot(x, y8, '-*', linewidth=2,alpha=.5, label = 'Xe')

    ax.grid(True)
    ax.legend(loc='upper left')
    plt.show()


x = [90, 60, 45, 30]
y1 = [0, 0, 0, 0]
y2 = [3.4, 7.0, 12.6, 25.4]
y3 = [31.8, 39.4, 52.6, 81.8]
y4 = [55.0, 66.2, 85.4, 128.6]

y5 = [0.05, 0.06, 0.07, 0.10]
y6 = [0.12, 0.14, 0.18, 0.25]
y7 = [0.28, 0.33, 0.40, 0.57]
y8 = [0.41, 0.48, 0.59, 0.83]

y5 = [2.1, 2.5, 3, 3.4]
y6 = [5.5, 6.4, 7.8, 11]
y7 = [12.6, 14.5, 17.8, 25.1]
y8 = [18.4, 21.2, 26.0, 36.8]

plot_2(y5,y6,y7,y8)
#plot_1(y1,y2,y3,y4)
