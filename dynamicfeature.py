from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import datashow as ds
import matplotlib.patches as mpatches
from pylab import rcParams
def peakdraw(xoutput):
    pointx = []
    pointy = []
    pointx1 = []
    pointy1 = []
    for x in range(len(xoutput[0])):
        pointx.append(xoutput[0][x][1])
        pointy.append(xoutput[0][x][0])
    for x in range(len(xoutput[1])):
        pointx1.append(xoutput[1][x][1])
        pointy1.append(xoutput[1][x][0])
    return pointx,pointy,pointx1,pointy1

filec = 'velocity_18.out'
sd1 = []
with open(filec, 'r') as f:
    for x in f:
        # print(x.strip())
        sd1.append(float(x.strip()))
        # sd.append([float(n) for n in x.strip().split(' ')])

# sd = np.array(ds.hjv)
numberofsample = int(len(sd1)/2)
sd = np.array(sd1[:8000])
print (len(sd))
kmeans = KMeans(n_clusters=2, random_state=None)
u = kmeans.fit(sd.reshape(-1, 1))
# print(u.cluster_centers_)
label = u.labels_
rcParams['ytick.labelsize'] = 15
rcParams['xtick.labelsize'] = 15
fig = plt.figure(figsize=(10,10))
xside = np.arange(len(sd))
colormap = np.array(['Red','Blue','Gold','Gray'])
markmap = np.array(['o','*'])
dotslabel =  np.array(['jogging','walking'])
# colormapprodicate = np.array(['Yellow','Green','Orange','Black'])
reddot = []
reddot1 =[]
bdot =[]
bdot1 =[]
reddot.append(xside[1])
reddot1.append(sd[1])
bdot.append(xside[22])
bdot1.append(sd[22])
plt.subplot(111)
for x, y1, l in zip(xside,sd,label):
    plt.scatter(x, y1, color=colormap[l],marker=markmap[l],s=40)
plt.scatter(reddot,reddot1,color=colormap[0],marker='o',label='walking')
plt.scatter(bdot,bdot1,color=colormap[1],marker='*',label='jogging')
# plt.title("Classify dynamic motion", fontsize=20)
plt.xlabel("Pulse Number",fontsize=20)
plt.ylabel("Relative Velocity", fontsize = 20)
# red_patch = mpatches.Patch(color='red', linestyle='o', label='Sit-to-stand')
# blue_patch = mpatches.Patch(color='blue', linestyle='*',label='Stand-to-sit')
plt.legend(scatterpoints = 1)
plt.xlim(0,len(sd))
plt.ylim(-0.4,4.5)
# cpo =peakdraw(ds.dryfx)
# y = np.linspace(0, len(ds.dryfy), len(ds.dryfy))

# ax1 = plt.subplot(121)
# # ax1.set_ylim([-2, 2])
# labels = ['0','10','20','30','40','50','60']
# ax1.plot(y, ds.dryfy)
# ax1.scatter(cpo[0],cpo[1],color = 'red')
# ax1.scatter(cpo[2],cpo[3],color = 'green')
# ax1.set_title("(a) Feature extraction",fontsize=20)
# ax1.set_xlabel("Time (s)",fontsize =20)
# ax1.set_xticklabels(labels,fontsize=10)
# ax1.set_ylabel("Relative distance (m)",fontsize=20)
# ax1.set_ylim([0,4])
plt.show()
fig.savefig('velcocity_test.png', bbox_inches='tight',format='png', dpi=100)
# fig.savefig('velcocity.png', bbox_inches='tight',format='png', dpi=100)