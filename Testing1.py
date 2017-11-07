from sklearn.cluster import KMeans
import functionclass as fc
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
from operator import itemgetter

# fc.yi = np.array(fc.yi)
# kmeans = KMeans(n_clusters=2,random_state=None)
# u= kmeans.fit(fc.yi.reshape(-1,1))
#
# print('u:',u.cluster_centers_)
# ulabel = u.labels_
# print('ulabel:',u.labels_)
#
# gh=[]
# for x in range(len(ulabel)-2):
#     if ulabel[x]-ulabel[x+1]!=0 and ulabel[x+1]==ulabel[x+2]:
#         gh.append(x)
# print(gh)
file1 = 'Datasit.out'
filetime1 = 'DataTimesit.out'

file = 'Datazhang.out'
filetime = 'DataTimezhang.out'
t0 = 9.765625000e-12
c =  299792458
signaldata = fc.readfile(file)
timedata = fc.readfile(filetime)

signaldata1 = fc.readfile(file1)
timedata1 = fc.readfile(filetime1)
#
# yi=fc.alldifferentpoint(signaldata)[1]
#
# print(yi)
# testd = fc.remove_nosie(yi)
# print(testd)
# disy = []
# for x in range(len(testd )):
#     disy.append((timedata[x][testd[x]]-t0)*c)

# fc.drawpicture(testd)
maxdifferenceloc = []
maxdifference=[]
difference=[]
op = 0
for t in range(len(signaldata)-1):
    for x in range(len(signaldata[1])):
        difference.append(float("%.2f" %(signaldata[t+1][x]-signaldata[t][x])))
    if np.max(difference) == 0:
        maxdifferenceloc.append(0)
        op +=1
    else:
        maxdifferencev = difference.index(np.max(difference))
        maxdifferenceloc.append(maxdifferencev)
    maxdifference.append(np.max(difference))
    difference=[]

print(maxdifferenceloc)
print(maxdifference)
print('number of zero: ',op,'/',len(maxdifference))
print('max divergent', np.max(maxdifference))
print('')
print('*************')
print('')

testingstanding1 = fc.alldifferentpoint(signaldata1)
print(testingstanding1[0])
print(testingstanding1[1])
print('number of zero: ',testingstanding1[2],'/',len(testingstanding1[0]))
print('max divergent', np.max(testingstanding1[1]))
# center = fc.getcluster_center(testd)
# print(center[0])
# print(fc.Getvelocity(signaldata,timedata,60))

yi = fc.convertmeters(maxdifferenceloc,timedata)
# fc.drawpicture(yi)
yisit = fc.convertmeters(testingstanding1[0],timedata1)
# fc.drawpicture(yisit)
featuresit = fc.gettheconsecutivelist(maxdifferenceloc)
featurecomplex = fc.gettheconsecutivelist(testingstanding1[0])

filter1 = fc.featurefilter(featuresit)
print(filter1)
increase = 0
decrease = 0
resultfeature = []
for y in range(len(filter1)):
    for x in range(len(filter1[y])-1):
        if maxdifferenceloc[filter1[y][x+1]]>=maxdifferenceloc[filter1[y][x]]:
            increase+=1
        else:
            decrease+=1
    rangefilter = maxdifferenceloc[filter1[y][-1]] - maxdifferenceloc[filter1[y][1]]
    resultfeature.append([increase,decrease,rangefilter])
    increase = 0
    decrease = 0
print(resultfeature)

filter2 = fc.featurefilter(featurecomplex)
print(filter2)
result2 = fc.featureresult(filter2,testingstanding1[0])
print(result2)


# for k,g in groupby(enumerate(indexlist), lambda ix: ix[0]-ix[1]):
#     print(itemgetter(1))
#     print(g)
# y1 = np.linspace(0, len(maxdifferenceloc), len(maxdifferenceloc))
# fig = plt.figure()
# ax1 = plt.subplot(211)
# # ax1.set_ylim([-2, 2])
# ax1.plot(y1, maxdifferenceloc)
# y2 = np.linspace(0, len(testingstanding1[0]), len(testingstanding1[0]))
# ax2 = plt.subplot(212)
# # ax1.set_ylim([-2, 2])
# ax2.plot(y2, testingstanding1[0])
# plt.show()
