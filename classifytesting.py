from sklearn.cluster import KMeans
import functionclass as fc
import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


filex = 'Data/Data.csv'

sd = []
with open(filex, 'r') as f:
    for x in f:
        # print(x.strip())
        sd.append(float(x.strip()))
        # sd.append([float(n) for n in x.strip().split(' ')])

print(sd)

sd = np.array(sd)
kmeans = KMeans(n_clusters=2, random_state=None)
u = kmeans.fit(sd.reshape(-1, 1))

iris = datasets.load_iris()
X = iris.data
print(X)

y = iris.target
print(y)
estimators = {'k_means_iris_3': KMeans(n_clusters=3),
              'k_means_iris_8': KMeans(n_clusters=8),
              'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
                                              init='random')}

fignum = 1
for name, est in estimators.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean() + 1.5,
              X[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
plt.show()

# file = 'Data/Datasit2.out'
# filetime = 'DataTimesit2.out'
# t0 = 9.765625000e-12
# c =  299792458
# signaldata = fc.readfile(file)
# timedata = fc.readfile(filetime)
#
# yti=fc.alldifferentpoint(signaldata)
# ymax = np.max(yti[1])
# print(ymax)


# disy = fc.convertmeters(yti[0],timedata)
# disy = fc.filterbasesavutzky(disy)
# output = fc.getcluster_center(2,disy)
# intervaltime = fc.interval_time(60, signaldata, timedata[1][-1])
# xoutput = fc.getxoutput(output,disy)
# velocitylist = fc.getVelocity(xoutput,intervaltime)


#
# print(u.cluster_centers_)
# p1 = np.array(yti[1]).reshape(-1,1)
# # p2 = np.array(velocitylist).reshape(-1,1)
# print(u.predict(p1))


# fc.addsaveamplituate(yti[1])