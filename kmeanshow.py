from sklearn.cluster import KMeans
import functionclass as fc
import numpy as np
import matplotlib.pyplot as plt


filex = 'Data/Data.csv'
filec = 'Data/Velocity.csv'
sd = []
with open(filec, 'r') as f:
    for x in f:
        # print(x.strip())
        sd.append(float(x.strip()))
        # sd.append([float(n) for n in x.strip().split(' ')])

print(sd)

sd = np.array(sd)
kmeans = KMeans(n_clusters=2, random_state=None)
u = kmeans.fit(sd.reshape(-1, 1))
print(u.cluster_centers_)
label = u.labels_
# print(label[:1000])
y = np.linspace(0, len(sd), len(sd))

file = 'Data/Dataliuwalk.out'
filetime = 'Data/DataTimeliuwalk.out'
t0 = 9.765625000e-12
c =  299792458
signaldata = fc.readfile(file)
timedata = fc.readfile(filetime)

yti=fc.alldifferentpoint(signaldata)
ymax = np.max(yti[1])
print(ymax)
p1 = np.array(yti[1]).reshape(-1,1)
labelt = u.predict(p1)
one = 0
for x in range(len(labelt)):
    if labelt[x] == 1:
        one+=1

print(labelt)
print(one,'/',len(labelt))
ky = np.linspace(len(sd),len(yti[1])+len(sd),len(yti[1]))

colormap = np.array(['Red','Blue','Gold','Gray'])
colormapprodicate = np.array(['Yellow','Green','Orange','Black'])

plt.scatter(y,sd,c=colormap[label],s=40)
plt.xlim(0,350)
# plt.scatter(ky,yti[1],c=colormapprodicate[labelt],s=40)
plt.show()