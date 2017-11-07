import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import functionclass as fc
import matplotlib.patches as mpatches
from pylab import rcParams

file = 'Data/Datazhusit1.out'
filetime = 'Data/DataTimezhusit1.out'
file1 = 'Data/Data1.out'
filetime1 = 'Data/DataTime1.out'
t0 = 9.765625000e-12
c =  299792458
signaldata = fc.readfile(file)
timedata = fc.readfile(filetime)

yti=fc.alldifferentpoint(signaldata)
yi = yti[0]

def drawfeaturecolor(colorlist,listdata,timedata):
    list =fc.convertmeters(listdata,timedata)
    y = np.linspace(0, len(list), len(list))
    rcParams['ytick.labelsize'] = 15
    rcParams['xtick.labelsize'] = 15
    fig = plt.figure(figsize= (11,8))
    plt.subplot(111)
    plt.plot(y, list, '+',color = 'y')
    for i in range(len(colorlist[0])):
        u1 = plt.plot(colorlist[0][i], list[colorlist[0][i][0]:(colorlist[0][i][-1] + 1)], '-r', color='r', label='Sit-to-stand')
    for i in range(len(colorlist[1])):
        u2 = plt.plot(colorlist[1][i], list[colorlist[1][i][0]:(colorlist[1][i][-1] + 1)], '-', color='b',label = 'Stand-to-sit')
    plt.ylim([0,4])
    labels = ['0', '10', '20', '30', '40', '50', '60']
    # plt.title(" Feature extract",fontsize=20)
    plt.xlabel("Time (s)",fontsize = 20)
    # plt.ylabel("Relative distance", fontsize=20)
    plt.xticks([i*150  for i, _ in enumerate(labels)], labels)
    red_patch = mpatches.Patch(color='red', label='Stand-to-sit')
    red_patch1 = mpatches.Patch(color='blue', label='Sit-to-stand')
    plt.legend(handles=[red_patch1,red_patch])


    # plt.subplot(122)
    # a = [len(colorlist[0]),len(colorlist[1])-1,0]
    # b = ['Sit-to-stand','Stand-to-sit','']
    # index = np.arange(len(a))
    # bar_width = 0.2
    # plt.bar(index+0.6,a,bar_width)
    # ax2.title("")
    # plt.title("(b) Statistics",fontsize=20)
    # plt.xlabel("Status",fontsize = 20)
    # plt.ylabel("Counts",fontsize = 20)
    # plt.ylim([0,10])
    # plt.xlim(0,1)
    # plt.xticks([i + 0.72 for i, _ in enumerate(b)], b)
    plt.show()
    fig.savefig('staticfeature.eps', format='eps', dpi=100)

statuslist = ['Sitting->Standing','Standing->Sitting','Walking','Running','Jumping']

if yti[2]>=(len(signaldata)*0.4) or np.max(yti[1])< 0.2:
    featurecomplex = fc.gettheconsecutivelist(yti[0])
    filter2 = fc.featurefilter(featurecomplex)
    # print(filter2)
    result2 = fc.featureresult(filter2, yti[0])
    print(result2)
    status = fc.verifystaticstatus(result2,yti[0])
    colorlist = fc.colorpoint(status)
    print(statuslist[0],': %d'%status[0])
    print(statuslist[1],': %d'%status[1])
    print('increased ',status[2])
    print('decreased ',status[3])
    print(len(colorlist[0]))
    print(len(colorlist[1]))
    drawfeaturecolor(colorlist,yti[0],timedata)
else:
    disy = fc.convertmeters(yi,timedata)
    disy = fc.filterbasesavutzky(disy)
    output = fc.getcluster_center(2,disy)
    intervaltime = fc.interval_time(60, signaldata, timedata[1][-1])
    xoutput = fc.getxoutput(output,disy)
    velocitylist = fc.getVelocity(xoutput,intervaltime)
    print(velocitylist)
    # velocity = fc.getVelocity(output,df,intervaltime)
    velocity = np.mean(velocitylist)
    status = statuslist[2]
    print('Status:',status)
    print('Velocity:', velocity)
