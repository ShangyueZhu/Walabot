import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import functionclass as fc
import csv
import pylab as pl
from pylab import rcParams
import matplotlib.image as mping

file = "Data/Amplitude.csv"
c = fc.readfile(file)
xdynamic = c[6][:230] #6

file = 'Data/Dataliuwalk.out'
filetime = 'Data/DataTimeliuwalk.out'
t0 = 9.765625000e-12
c =  299792458
signaldata = fc.readfile(file)
timedata = fc.readfile(filetime)

samplenumber = 140
signaldataw = np.array(signaldata)[:samplenumber].reshape(-1,1)
timedata1 = np.array(timedata)[0][0]
timedatalast = np.array(timedata)[0][-1]
# for x in range(0,len(timedata[0]),1000):
    # print timedata[0][x]

rcParams['ytick.labelsize'] = 15

y = np.linspace(0, len(signaldataw), len(signaldataw))
y1 = np.linspace(0,len(signaldata[0]),len(signaldata[0]))
y2 = np.linspace(0,len(xdynamic),len(xdynamic))
divergent = np.array(signaldata[1]) - np.array(signaldata[0])
labels = ['0','25','50','75','100','125','150']
# fixlabels = ['0','6','18','30','36']
Timelabels = ['0','','4000','','8000','','12000','','16000']
# Timefixlabels = ['0','9.76e-09','2.92e-08','4.88e-08','6.84e-08']
fig = plt.figure(figsize=(12,8))

s = []
for x in range(len(signaldataw)+1):
    if x%((14*8192))==0:
        s.append(-0.6)
    else:
        s.append(0.5)
ys = np.arange(len(s))



# ax1 = plt.subplot(111)
# ax1.set_ylim([-1.5, 1.5])
# ax1.plot(y,signaldataw)
# ax1.plot(ys,s,'r--')
# ax1.annotate('Long time window (window size:14)', xy=(14*3.5*8192, 0.45), xytext=(14*4*8192, 0.54),
#             arrowprops=dict(facecolor='black', shrink=0.01),fontsize=15)
# ax1.text(14*1.5*8192, 0.51, "14",fontsize=15)
# ax1.set_title("(a) Long time",fontsize=20)
# ax1.set_xlabel('Number of pulse \n',fontsize=18)
# ax1.set_ylabel('Amplitude',fontsize=20)
# print('asdf')
# ax1.set_xticklabels(labels,fontsize=15)
# ax1.set_ylim([-0.6, 0.6])
#
twopulse = []
for x in range(0,40,20):
    for y in range(len(signaldata[x])):
        twopulse.append(signaldata[x][y])
print ("twopulse: ",len(twopulse))
d = []
for x in range(len(signaldata[0])+1):
    if x%(len(signaldata[0])/16)==0:
        d.append(-0.408)
    else:
        d.append(0.33)
# print d
yd = np.arange(len(d))

ls = []
for x in range(len(twopulse)+1):
    if x % (8192) == 0:
        ls.append(0.43)
    else:
        ls.append(0.5)
yls = np.arange(len(ls))
# print('next')
ax2 = plt.subplot(111)
# ax2.set_ylim([-1.5, 1.5])
ytwo = np.arange(len(twopulse))
ax2.plot(ytwo,twopulse)
# ax2.plot(y1,signaldata[0])#signaldata[0]
ax2.plot(yd,d,'r--')
ax2.plot(yls,ls,'b--')
ax2.annotate('Short window (window size:500)', xy=(2252, 0.26), xytext=(3003, 0.34),
            arrowprops=dict(facecolor='black', shrink=0.01),fontsize=15)
ax2.annotate('', xy=(2003, 0.5), xytext=(2003, 0.43),
            arrowprops=dict(facecolor='black', shrink=0.01),fontsize=15)
ax2.annotate('', xy=(8193, 0.52), xytext=(9103, 0.52),
            arrowprops=dict(facecolor='black', shrink=0.01),fontsize=10)
ax2.annotate('', xy=(16384, 0.52), xytext=(15603, 0.52),
            arrowprops=dict(facecolor='black', shrink=0.01),fontsize=10)
ax2.text(200, 0.4, "Long window (window size: 16 short windows)",fontsize=15)
ax2.text(10200, 0.4, "Long window ",fontsize=15)
ax2.text(9600, 0.52, "Pulse Repetition Interval(PRI)",fontsize=15)
# ax2.set_title(" Short time",fontsize=20)
ax2.set_xlabel('Number of sample',fontsize=18)
ax2.set_ylabel('Amplitude',fontsize=20)
ax2.set_xticklabels(Timelabels,fontsize=15)
ax2.set_ylim([-0.4, 0.65])

# ax3 = plt.subplot(111)
# # ax3.set_ylim([-1.5, 1.5])
# yd = np.array(xdynamic)/450
# xf = np.fft.fft(yd)
# print (xf[:80])
# ax3.plot(y2,yd)
# ax3.set_title("Motion information",fontsize=20)
# ax3.set_xlabel('Number of pulse',fontsize=20)
# ax3.set_ylabel('Amplitude',fontsize=20)
# ax3.set_xticklabels(Timelabels,fontsize=15)
fig.savefig('LT.png', bbox_inches='tight',format='png', dpi=450)

plt.show()
def drawpicture(list):
    y = np.linspace(0, len(list), len(list))
    fig = plt.figure()
    ax1 = plt.subplot()
    ax1.set_ylim([-1.5, 1.5])
    ax1.plot(y,list)
    plt.show()
    # fig.savefig("slow_time")

# drawpicture(signaldataw)


# x = np.linspace(0, len(yi), len(yi))
# localregression = fc.lowess(x,yi,f=0.25,iter=3)
# pl.clf()
# pl.plot(x,localregression)
# pl.show()
# h = np.array([0,234,678,322,865,546])
# disy = np.array(disy)
# kmeans = KMeans(n_clusters=3, random_state=None)
# u = kmeans.fit(h.reshape(-1, 1))
# print(u)
# h1 = np.array([[0],[456],[123],[322],[843],[546]])
# h2 = np.array([0,234,678,322,865,546])
#
# t = u.cluster_centers_.shape[1]
# eq=h2.reshape(-1,1)
# print(u.predict(eq))
# print(t)
# file = 'Data2.out'
#
# sd = []
# with open(file,'r') as f:
#     for x in f:
#         sd.append([float(n) for n in x.strip().split(',')])
#
# sd= np.reshape(sd,-1)
# y = np.linspace(0,len(sd),len(sd))
# fig = plt.figure()
#
# ax1 = plt.subplot()
# ax1.set_ylim([-5, 5])
# ax1.plot(y,sd)
# plt.show()

# with open(file,'r') as f:
#     for x in f:
#         sd.append([float(n) for n in x.strip().split(',')])
# print(len(totoall))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# q = 0
# y = np.linspace(0,8192,8192)
# u=0
#
# for x in range(len(sd)):
#     ax.plot(y,sd[x],zs = u,zdir = 'x')
#     u = u+1
# # for x in range(1, len(signallist)):
# #     if x % 8192 == 0:
# #         # print(signallist[q:x])
# #         ax.plot( y,signallist[q:x], zs=u, zdir='x')
# #         q = x
# #         u  = u + 1
# print(u)
# ax.legend()
# ax.set_xlim(0, len(sd)+1)
# ax.set_ylim(0, 8200)
# ax.set_zlim(-2, 2)
# ax.set_xlabel('Slow Time')
# ax.set_ylabel('Fast Time')
# ax.set_zlabel('Amplitude')
# ax.view_init(elev=12., azim=-75)
# fig.tight_layout(rect=[-0.11,-0.13,1.09,1.21])
# plt.show()