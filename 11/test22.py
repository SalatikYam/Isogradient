import numpy as np
import math
import matplotlib.pyplot as plt

class element(object):
    def __init__(self, name, x, y):
        #self.id = id
        self.x = x
        self.y = y
        self.name = name
    def inf(self):
        print("\n  "+self.name+"\n("+str(self.x)+", "+str(self.y)+")\n")

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

def fill_zeros(v,h):
    el_zero = [ [element("Z0", 0, 0)]*h for i in range(v)]
    return el_zero

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

def el_count(el):
    el_name=el_list(el)
    el_name_count = np.zeros(len(el_name))
    for i in range(len(el)):
        for j in range(len(el_name)):
            if el_name[j]==el[i].name:
                el_name_count[j] = el_name_count[j] + 1
    return el_name_count

def r_from_fi(el, fi):
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
    metrick = np.zeros(len(el_name))
    for i in range(len(el)):
        for j in range(len(el_name)):
            if el_name[j]==el[i].name:
                metrick[j] = metrick[j]+abs(a*(el[i].x+0.5)+b*(el[i].y-0.5)+c)/(R*math.sqrt(a**2+b**2))    
    el_name_count = el_count(el)
    metrick= metrick/el_name_count
    return metrick
    
    #d = abs(a*el[1].x+b*el[1].y+c)/math.sqrt(a**2+b**2)
    #print(d)
    #print(el[0].x,el[0].y,C[0])
    #print(np.linalg.norm(center))

def plot_graph(el):
    pi = math.pi
    el_name = el_list(el)
    r = [ []*1 for i in range(len(el_name))]
    dr = [ []*1 for i in range(len(el_name))]
    FI = np.arange(0, 2*pi, pi/180)
    for fi in FI:
        for i in range(len(el_name)):
            r[i].append(r_from_fi(el, fi)[i])
            dr[i].append(abs(r_from_fi(el, fi)[i]-1)*pi/180)
    for i in range(len(el_name)):
        dr[i] = sum(dr[i])
    ax = plt.subplot( projection='polar')
    ax.set_rlabel_position(0)
    ax.plot(FI, np.ones(len(FI)), '--', linewidth=2,color='green',alpha=.5, label = 'ideal')
    ax.fill_between(FI, np.ones(len(FI)),color='g',alpha=.05)
    for i in range(len(el_name)):
        ax.plot(FI, r[i], linewidth=4,alpha=.6, label = el_name[i]+" "+str(round(dr[i],1)))
    ax.grid(True)
    el_name_count = el_count(el)
    Metrick = sum(dr*el_name_count)/sum(el_name_count)
    print(Metrick)
    ax.legend(loc='upper left')
    plt.show()



el = read_elements("massif2.txt")
plot_graph(el)
"""
ax = plt.subplot(111,projection='polar')
ax.set_rlabel_position(0)
ax.plot(FI, np.ones(len(FI)), '--', linewidth=2,color='green',alpha=.5, label = 'ideal')
ax.plot(FI, r, linewidth=3,color='red',alpha=.5, label = 'A0')
ax.fill_between(FI, np.ones(len(FI)),color='g',alpha=.2)
#ax.fill_between(FI, r,color='r',alpha=.2)
ax.grid(True)
ax.legend()
plt.show()
#r_from_fi(el,0)
#print(np.arange(0, 2*pi, pi/10))
#el_print(el)
#el_names = el_list(el)
#print(el_names)

#print(el[1].inf())
"""

#read_elements("massif.txt")
"""
f = open("massif.txt")
text = f.readlines()
temp = text[0].replace('\n', '')
temp = temp.split(sep = ' ')
#temp = re.split(' |\n', text[0])
print(temp)

el = [[],[]]
for i in range(10):
    el[0].append(element(0,i,0))
for i in range(5):
    el[1].append(element(1,i,1))
print(len(el[0]))
for j in range(len(el)):
    for i in range(len(el[j])):
        el[j][i].inf()
#for i in range(10):
#    el.append(element(0,i,0))
#for i in range(len(el)):
#    el[i].inf()
"""
"""
def printing(el):
    y = []
    for i in range(len(el)):
        y.append(el[i].y)
    ymin = min(y)
    el_new = []
    for i in range(-min(y)+1):
        el_temp = []
        el_temp1 = []
        for j in range(len(el)):
            if el[j].y == -i:
                el_temp.append(el[j])
        for j in range(len(el_temp)):
            for k in range(len(el_temp)):
                if el_temp[k].x == j:
                    el_temp1.append(el_temp[k])
        #print(len(el_temp))
        el_new.append(el_temp1)
"""