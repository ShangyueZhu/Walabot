from sklearn.cluster import KMeans
import functionclass as fc
import numpy as np
import matplotlib.pyplot as plt
import csv

file = 'Data/Dataliuwalk.out'
filetime = 'Data/DataTimeliuwalk.out'
t0 = 9.765625000e-12
c =  299792458
signaldata = fc.readfile(file)
timedata = fc.readfile(filetime)

yti=fc.alldifferentpoint(signaldata)
yi = yti[0]
# print (yi,len(yi))


ymax = np.max(yti[1])
# print(yti[1])
# print(ymax,type(yti[1][0]))
# print(len(yti[1]))

# fc.addsaveamplituate(yti[1])

disy = []
for x in range(len(yi)):
    disy.append((timedata[x][yi[x]]-t0)*c)
# print disy

# print(disy,type(disy))

# g = np.min(disy[26:48])
# inssad = disy.index(g)
# print(g)
# print(inssad)
# fc.drawpicture(disy)

disy = fc.filterbasesavutzky(disy)

output = fc.getcluster_center(2,disy)
# print('center: ',output[0])
# print('output: ',output[1])
df =fc.getClassificationpoint(output[1])
# print(df)

intervaltime = fc.interval_time(60,signaldata,timedata[1][-1])

# print('intervaltime: ',intervaltime)
hjv = []
for x in range(len(disy)-1):

    hjv.append(np.abs(disy[x+1]-disy[x])/intervaltime)

# print (hjv,len(hjv))


cv = []
for x in range(len(df)-1):
    if df[x+1] - df[x] >= 10:
        cv.append(df[x])
cv.append(len(output[1])-1)
if cv[0]<5:
    cv.remove(cv[0])
# print(cv)
# xoutput = fc.peakdetectioncrestfrist(cv,disy)
# xoutput = fc.peakdetectiontroughfirst(cv,disy)

if output[0][0]>output[0][1] and output[1][cv[0]-1]==1 :
    xoutput = fc.peakdetectioncrestfrist(cv,disy)
elif output[0][0]< output[0][1] and output[1][cv[0]-1]==0 :
    xoutput = fc.peakdetectioncrestfrist(cv, disy)
else:
    xoutput = fc.peakdetectiontroughfirst(cv,disy)



if xoutput[1][0][1]>xoutput[0][1][1]:
    xoutput[0].remove(xoutput[0][0])
    # print('Removed a mistake')
# xoutput[0].remove(xoutput[0][0])
# print(xoutput[0])
# print(xoutput[1])
# print(len(xoutput[0]),len(xoutput[1]))

mark1 = False
mark0 = False
markq = False
velcocitylisto = []
if len(xoutput[1])>len(xoutput[0]):
    a = len(xoutput[0])
    mark0 = True
elif len(xoutput[1])==len(xoutput[0]):
    a = len(xoutput[1])
    markq = True
else:
    a = len(xoutput[1])
    mark1 = True
# print(a)
for x in range(a):
    # if x+1 > a-1:
    distenceF = np.abs(xoutput[0][x][0] - xoutput[1][x][0])
    timeF = (np.abs(xoutput[1][x][1] - xoutput[0][x][1]))*intervaltime
    velcocitylisto.append(distenceF/(timeF/2))
    if markq == True:
        if x + 1 < a:
            distenceF = np.abs(xoutput[0][x + 1][0] - xoutput[1][x][0])
            timeF = (np.abs(xoutput[1][x][1] - xoutput[0][x + 1][1])) * intervaltime
            velcocitylisto.append(distenceF / (timeF / 2))
    if mark1 ==True:
        distenceF = np.abs(xoutput[0][x + 1][0] - xoutput[1][x][0])
        timeF = (np.abs(xoutput[1][x][1] - xoutput[0][x + 1][1])) * intervaltime
        velcocitylisto.append(distenceF / (timeF / 2))
    if mark0 ==True:
        distenceF = np.abs(xoutput[0][x][0] - xoutput[1][x+1][0])
        timeF = (np.abs(xoutput[1][x+1][1] - xoutput[0][x][1])) * intervaltime
        velcocitylisto.append(distenceF / (timeF / 2))

# print('velocity: ',velcocitylisto)
# print('average: ',np.mean(velcocitylisto))
# print(len(velcocitylisto))


i0=0
for x in range(len(velcocitylisto)):
    if velcocitylisto[x-i0]<0.3:
        velcocitylisto.remove(velcocitylisto[x-i0])
        i0+=1
# print('velocity: ', velcocitylisto)

# fc.addsaveVelocity(velcocitylisto)
dryfx = xoutput
dryfy = disy
# fc.peakdraw(xoutput,disy)



# p = fc.getintervaldifferent(cv)
# print('intervaldifferent: ',p)
# velocitylist = []
# for x in range(len(p)):
#     timeforsingle = float((p[x]*intervaltime)/2)
#     distanceforsingle = np.max(output[0])-np.min(output[0])
#     v = distanceforsingle/timeforsingle
#     velocitylist.append(v)
# print('velocity: ',velocitylist)
# average = np.mean(p)
# print('averageintervalpoint: ',average)

# velocity = (np.max(output[0])-np.min(output[0]))/((average*intervaltime)/2)
# print('velocity: ',velocity)
# fc.drawpicture(testy)
# print(fc.getVelocity(signaldata,timedata,60))


def peakdraw(xoutput,disy):
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
    y = np.linspace(0, len(disy), len(disy))
    fig = plt.figure()
    ax1 = plt.subplot()
    # ax1.set_ylim([-2, 2])
    ax1.plot(y, disy,)
    ax1.scatter(pointx,pointy,color = 'red')
    ax1.scatter(pointx1,pointy1,color = 'green')
    plt.show()

