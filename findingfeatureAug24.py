import numpy as np
import functionclass as fc
import matplotlib.pyplot as plt
# import cv2

file = 'Data/Datastanding.out'
filetime = 'Data/DataTimestanding.out'
t0 = 9.765625000e-12
c =  299792458
signaldata = fc.readfile(file)
timedata = fc.readfile(filetime)

# print(len(signaldata))
t8192 = timedata[0][-1]
# print((t8192-t0)/8192)
testlength = int(len(signaldata)/20)

def addsaveA(list):
    with open('Data/Amplitude.csv', 'a') as f:
        for x in range(len(list)):
            output_string = str('%.1f' % list[x])
            output_string += ","
            f.writelines(output_string)
        f.write('\n')
    f.close()
def addsavelabel(list):
    with open('Data/label.csv', 'a') as f:
        for x in range(len(list)):
            output_string = str('%.1f' % list[x])
            output_string += " "
            f.writelines(output_string)
        f.write('\n')
    f.close()

def getreshape(orignaldifference,shape):
    lenthgh = int(len(orignaldifference)/shape)
    # print(lenthgh)
    neworginal = []
    for x in range(lenthgh*shape):
        neworginal.append(orignaldifference[x])
    neworginal = np.array(neworginal)
    nxor = np.reshape(neworginal,(int(len(neworginal)/shape),8192,shape))
    # print(nxor)
    return nxor

# testing2 = getreshape(signaldata,20)
# print(testing2)
# print(testing2.shape)

difference = []
maxdifferenceloc = []
maxdifference = []
orignaldifference = []
orignaldifference1 = []
maxdifferencev=0
op = 0
for y in range(1,len(signaldata)):
    for x in range(len(signaldata[1])):
        difference.append(float("%.2f" % (signaldata[y][x] - signaldata[y-1][x])))
    if np.max(difference) == 0:
        maxdifferenceloc.append(0)
        op+=1
    else:
        maxdifferencev = difference.index(np.max(difference))
        maxdifferenceloc.append(maxdifferencev)
    maxdifference.append(np.max(difference))
    orignaldifference.append(difference)
    # orignaldifference1.append(signaldata[y-1][maxdifferencev])
    difference=[]

print(len(orignaldifference))
# orxxxx = np.reshape(np.array(orignaldifference),(len(orignaldifference),8192))


# print(len(orignaldifference))
print("............")
print(len(orignaldifference[0]))
print("............")
print(maxdifferenceloc)
print("............")
print(maxdifference)

fc.drawpicture(maxdifferenceloc)
ordiffloc = []
for x in range(int(len(maxdifferenceloc)/20)*20):
    ordiffloc.append(maxdifferenceloc[x])


lable = [1,0]
# addsaveA(ordiffloc)
# addsavelabel(lable)
def testingspectrum(ordiffloc):
    ordiffloc = np.array(ordiffloc).reshape(len(ordiffloc) / 20, 20)
    TY = len(ordiffloc)
    i = np.empty((TY, 20))
    for x in range(20):
        sing = np.array(ordiffloc[x])
        a = np.fft.fft(sing, TY)
        b = abs(a)
        z = b.reshape(TY, 1)
        i[:, x] = z.flatten()
    print(i[:,1].shape)
    fig = plt.figure(frameon=False)
    plt.imshow(i, cmap='cool', interpolation='nearest')
    # plt.gca().axis('off')
    plt.show()
    print(i.shape)
    # cv2.imwrite('Image/sitting11.png', i)

# testingspectrum(ordiffloc)
# print("nil lenth: ",len(ordiffloc)/20)

# orxxxx = np.reshape(np.array(ordiffloc),(len(ordiffloc)/20,20))
#
# print(orxxxx)
# print(orxxxx.shape)
#
#
# ordiffloc = np.array(ordiffloc).reshape(len(ordiffloc)/20,20)
# TY = len(ordiffloc)
# i = np.empty((TY,20))
#
# print(TY)
# for x in range(20):
#     sing = np.array(ordiffloc[x])
#     a = np.fft.fft(sing,TY)
#     b = abs(a)
#     z = b.reshape(TY,1)
#     i[:,x] = z.flatten()
#
# print(i)
# #
# plt.imshow(i,cmap='cool',interpolation='nearest')
# plt.show()


# def readtxt():
#     name = 'Data/Amplitude.txt'
#     array = np.loadtxt(name)
#     print(array.reshape(8192, 20, -1).shape)
#
# readtxt()

# def addsaveA(list):
#     list = np.reshape(list, (8192*20, -1))
#     np.savetxt('Data/Amplitude.txt',  list.astype(np.float32))


def drawpicture(list):
    y = np.linspace(0, len(list), len(list))
    fig = plt.figure()
    ax1 = plt.subplot()
    ax1.set_ylim([-2, 2])
    ax1.plot(y, list)
    plt.show()

# addsaveA(testing2)
# drawpicture(orxxxx[155])

