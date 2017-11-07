import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz
from scipy import signal

def selectsample(p):
    b=[]
    bx = []
    for x in range(p.shape[0]):
        b.append([p[x][6],p[x][19],p[x][33], p[x][56]])
        bx.append([x*82+6,x*82+19,x*82+33,x*82+56])
    b = np.array(b).reshape(-1)
    bx = np.array(bx).reshape(-1)
    return b,bx
def removezeros(b):
    b1 = np.array(b[0]).reshape(-1)
    b2 = np.array(b[1]).reshape(-1)
    c = []
    c1 = []
    for x in range(b1.shape[0]):
        if b1[x] > 0.1:
            c.append(b1[x])
            c1.append(b2[x])
    c = np.array(c)
    c1 = np.array(c1)
    return c,c1
def butter_bandpass(highfreq, fs, order=5):
    nyq = 0.5 * fs
    #low = lowcut / nyq
    high = highfreq / nyq
    b, a = butter(order, Wn=high, btype='high')
    return b, a

def butter_bandpass_filter(data, highfreq, fs, order=5):
    b, a = butter_bandpass(highfreq=highfreq, fs=fs, order=order)
    y = lfilter(b, a, data)
    return y

def readfile(file):
    sd=[]
    with open(file,'r') as f:
        for x in f:
            sd.append([float(n) for n in x.strip().split(',')])
    return sd

file = "Amplitude.csv"
c = readfile(file)
y = c[13] #13
y1 = c[6] #6
#setlow = 20
minlen = np.min([len(y),len(y1)])
int(minlen/10)
samplingy = np.reshape(np.array(y[:820]),(10,82))
samplingy1 = np.reshape(np.array(y1[:820]),(10,82))
sethigh = 30

# print(samplingy.shape)
# print(samplingy1.shape)
v =selectsample(samplingy)
v1 =selectsample(samplingy1)
selectsample1 = removezeros(v)
selectsample2 = removezeros(selectsample(samplingy1))
# print('v1[0]',v1[0])
# print('v1[1]',v1[1])
# print(selectsample1[0])
# print(selectsample1[1])

frequencyofsit = len(selectsample1[0])/len(v[0])
frequencyofwalk = len(selectsample2[0])/len(v1[0])
# print(frequencyofsit)
# print('frequencywalk: ',frequencyofwalk)
# fig = plt.figure(figsize=(8,12))

# labels = ['0', '10', '20', '30', '40', '50', '60']
# t=np.linspace(0,fs,fs)
# ax1yset = [float(n/450) for n in y]
# plt.subplot(211)
# plt.plot(t,ax1yset)
# plt.title("(a) Static motion",fontsize=20)
# plt.ylabel("Relative Distance (m)",fontsize=20)
# plt.xlabel("Time(s)",fontsize=15)
# plt.xticks([i * 150 for i, _ in enumerate(labels)], labels)
# ax2 = plt.subplot(232)
#
# ax2.scatter(v[1],v[0]/750)
# ax2.set_ylim([0,1.5])
# ax2.set_xlim([0,850])

# fs1= len(y1)
# print(fs1)
# Ts1=1/fs1
# t1=np.linspace(0,fs1,fs1)
# ax3yset = [float(n/300) for n in y1]
# plt.subplot(212)
# plt.plot(t1,ax3yset)
# plt.title("(b) Dymatic motion",fontsize=20)
# plt.ylabel("Relative Distance(m)",fontsize=20)
# plt.xlabel("Time (s)",fontsize=15)
# plt.xticks([i * 150 for i, _ in enumerate(labels)], labels)
# fig.savefig('motion.pdf', bbox_inches='tight',format='pdf', dpi=400)

# fig = plt.figure(figsize=(8,12))
# plt.subplot(211)
# markerline, stemlines, baseline = plt.stem(v[1], v[0]/450, '-.')
# plt.setp(baseline, 'color', 'r', 'linewidth', 2)
# labels = ['0', '10', '20', '30', '40', '50', '60']
#
# plt.title("(c) Static Sampling",fontsize=20)
# plt.xlabel("Time (s)", fontsize=15)
# plt.ylabel("Relative distance (m)", fontsize=20)
# plt.xticks([i * 150 for i, _ in enumerate(labels)], labels)
# # plt.yticks([i/3 + 0.05 for i, _ in enumerate(ylabels)], ylabels)
#
#
# plt.subplot(212)
# markerline1, stemlines1, baseline1 = plt.stem(v1[1], v1[0]/300, '-.')
# plt.setp(baseline1, 'color', 'r', 'linewidth', 2)
# plt.title("(d) Dynamic Sampling",fontsize=20)
# plt.xlabel("Time (s)", fontsize=15)
# plt.ylabel("Relative distance (m)", fontsize=20)
# plt.xticks([i * 150 for i, _ in enumerate(labels)], labels)
# fig.savefig('sampling.pdf', bbox_inches='tight',format='pdf', dpi=400)

# fig = plt.figure(figsize=(13,8))
# fs= len(y)
# print(fs)
# Ts=1/fs
# t=np.linspace(0,fs,fs)
# ax1yset = [float(n/750) for n in y]
# ax1= plt.subplot(231)
# ax1.plot(t,ax1yset)
# ax1.set_title("(a) Static motion",fontsize=20)
# ax1.set_ylabel("Relative Distance",fontsize=20)
# ax1.set_xlabel("Time",fontsize=20)
#
# yf=np.fft.fft(y)
# xf=np.fft.fftfreq(len(yf),Ts)
# # yt = butter_bandpass_filter(y,20,40,fs, order=5)
# opvmax = np.max(yf)
# opvmin = np.min(yf)
# print(np.max(yf))
# print(np.min(yf))
# print(np.max(yf)-np.min(yf))
# print(np.max([np.abs(opvmax/opvmin),np.abs(opvmin/opvmax)]))
# ax2 = plt.subplot(232)
# # ax2yset = [float(n/5000) for n in yf]
# ax2.plot(xf,np.abs(yf))
# ax2.set_xlim([-50,50])
# ax2.set_title("(c) motion",fontsize=20)
# ax2.set_ylabel("Frequency",fontsize=20)
# ax2.set_xlabel("Time",fontsize=20)
# # ax2.set_ylim([-50,150000])
#
fs1= len(y1)
# print(fs1)
Ts1=1/fs1
t1=np.linspace(0,fs1,fs1)
ax3yset = [float(n/750) for n in y1]
ax3= plt.subplot(234)
ax3.plot(t1,ax3yset)
ax3.set_title("(b) Dymatic motion",fontsize=20)
ax3.set_ylabel("Relative Distance",fontsize=20)
ax3.set_xlabel("Time",fontsize=20)
#
yf1=np.fft.fft(y1)
xf1=np.fft.fftfreq(len(yf1),Ts1)
print(yf1[:50])
# opvmax1 = np.max(yf1)
# opvmin1 = np.min(yf1)
# print(np.max(yf1))
# print(np.min(yf1))
# print(np.max(yf1)-np.min(yf1))
# print(np.max([np.abs(opvmax1/opvmin1),np.abs(opvmin1/opvmax1)]))
# ax4yset = [float(n/5000) for n in yf1]
# ax4 = plt.subplot(235)
# ax4.plot(xf1,np.abs(ax4yset))
# ax4.set_xlim([-50,50])
# ax4.set_title("(d) motion",fontsize=20)
# ax4.set_ylabel("Frequency",fontsize=20)
# ax4.set_xlabel("Frequency",fontsize=20)
# ax4.set_ylim([-50,200000])
#


# ax5 = plt.subplot(211)
# b,a=butter_bandpass(sethigh,fs,5)

# yt=butter_bandpass_filter(y,sethigh,fs,5)
# xf=np.fft.fftfreq(len(yt),Ts)
# fs= len(v[0])
# Ts=1/fs
# yf2=np.fft.fft(samplingy)
# xf2=np.fft.fftfreq(len(yf2),Ts)
# ax5.plot(xf2,np.abs(yf2))
# # ax5.plot(xf,np.abs(ax2yset))
# # ax5.set_xlim([-50,50])
#
# ax6 = plt.subplot(212)
# fs1= len(v1[0])
# Ts1=1/fs1
# # b,a=butter_bandpass(setlow,sethigh,fs1,5)
# # yt1=butter_bandpass_filter(y1,sethigh,fs1,5)
# # xf1=np.fft.fftfreq(len(yt1),Ts1)
# yf3=np.fft.fft(samplingy1)
# xf3=np.fft.fftfreq(len(yf3),Ts1)
# # ax6.plot(xf1,np.abs(ax4yset))
# ax6.plot(xf3,np.abs(yf3))
# ax6.set_xlim([-50,50])
#
# plt.show()