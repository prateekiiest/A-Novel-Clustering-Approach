## importing all the modules
import seaborn as sns
import numpy as np
import sklearn.datasets as dm
import matplotlib.pyplot as plt


raw_data = dm.load_iris()
max_col = np.max(raw_data.data,axis=0)
min_col = np.min(raw_data.data,axis=0)

data = raw_data.data - min_col
data = np.divide(data,max_col-min_col)


## Dice Similarity
def dice(ti,tj):
    prod = 2*np.sum(ti*tj)
    length = np.sum(ti*ti)+np.sum(tj*tj)
    return prod/length
    
dice_sim = np.zeros([150,150])
for i in range(150):
    for j in range(150):
        dice_sim[i][j] = dice(data[i],data[j])
        
        
## Jacard Similarity
def jaccard(ti,tj):
    prod = np.sum(ti*tj)
    length = np.sum(ti*ti)+np.sum(tj*tj)-prod
    return prod/length
    
jac_sim = np.zeros([150,150])
for i in range(150):
    for j in range(150):
        jac_sim[i][j] = jaccard(data[i],data[j])
        
        
        
        
## Cosine Similarity
from math import sqrt
def cosine(ti,tj):
    prod = np.sum(ti*tj)
    length = sqrt(np.sum(ti*ti))*sqrt(np.sum(tj*tj))
    return prod/length
    
cos_sim = np.zeros([150,150])
for i in range(150):
    for j in range(150):
        cos_sim[i][j] = cosine(data[i],data[j])
        
        
        
## Overlap Similarity
def overlap(ti,tj):
    prod = (np.sum(ti*tj))
    length = min(np.sum(ti*ti),np.sum(tj*tj))
    return prod/length
    
over_sim = np.zeros([150,150])
for i in range(150):
    for j in range(150):
        over_sim[i][j] = overlap(data[i],data[j])
        


## MaxMin

def maxmin(sim):
    sim_1 = np.zeros([150,150])
    for r in range(150):
        avg = (np.sum(sim[r])-sim[r][r])/149
        for c in range(150):
            if sim[r][c]>avg:
                sim_1[r][c] = 1
    return sim_1
    

## Calculating all similarities
dice_sim_1 = maxmin(dice_sim)
jac_sim_1 = maxmin(jac_sim)
cos_sim_1 = maxmin(cos_sim)
over_sim_1 = maxmin(over_sim)


def grouping(sim):
    group = []
    for r in range(150):
        group.append((np.flatnonzero(sim[r]==1)))
    return group
    
dice_gr = grouping(dice_sim_1)
jac_gr = grouping(jac_sim_1)
cos_gr = grouping(cos_sim_1)
over_gr = grouping(over_sim_1)


def cl_mat(group):
    n_cl = len(group)
    cluster_mat = np.zeros([n_cl,n_cl])
    for i in range(n_cl):
        for j in range(n_cl):
            union = np.union1d(group[i],group[j])
            inter = np.intersect1d(group[i],group[j])
            cluster_mat[i][j] = inter.shape[0]/union.shape[0]
            if i==j:
                cluster_mat[i][j] = 0
    return cluster_mat
    
    
def Hierachial(group,n_cl=150,f_cl=3):
    temp_gr = list(group)
    count = 0
    while n_cl>f_cl:
        cluster_mat = cl_mat(temp_gr)
        maxe = np.max(cluster_mat)
        max_i = np.argwhere(cluster_mat==maxe)
        merge_cl = []
        delete_i = np.array([])
        rand = np.random.randint(0,max_i.shape[0]-1)
        m_cl = np.union1d(temp_gr[max_i[rand][0]],temp_gr[max_i[rand][1]])
        merge_cl.append(m_cl)
        for row in range(n_cl):
            if np.array_equal(np.intersect1d(m_cl,temp_gr[row]),temp_gr[row]):
                delete_i = np.union1d(delete_i,np.array([row]))
        temp_gr = [i for j, i in enumerate(temp_gr) if j not in delete_i]
        for i in range(len(merge_cl)):
            temp_gr.append(merge_cl[i])
        n_cl = len(temp_gr)
        print('Iter '+str(count)+'  n_cl = '+str(n_cl))
        count = count+1
    return temp_gr,n_cl
    
    
    
group,n_cl = Hierachial(dice_gr,150)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2,random_state=1)
pca.fit(data)
X_new = pca.transform(data)

plt.figure(figsize=(25,7))
color1 = ['w' for x in range(150)]
for i in group[0]:
    color1[i] = 'r'
plt.subplot(131)
plt.scatter(X_new[:,0],X_new[:,1],color=color1)

color2 = ['w' for x in range(150)]
for i in group[1]:
    color2[i] = 'g'

plt.subplot(132)
plt.scatter(X_new[:,0],X_new[:,1],color=color2)

color3 = ['w' for x in range(150)]
for i in group[2]:
    color3[i] = 'b'
    
plt.subplot(133)
plt.scatter(X_new[:,0],X_new[:,1],color=color3)
plt.show()

