import random
import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def Average(lst):
    Average = int(sum(lst) / len(lst))
    return Average


def simpler(Fish_Number_List):
    simp = []
    x = 16
    simp.append(Average(Fish_Number_List[:x]))
    while 1:
        try:
            simp.append(Average(Fish_Number_List[x:x+30]))
            x += 30
        except:
            if len(Fish_Number_List[x:]) > 0:
                simp.append(Average(Fish_Number_List[x:]))
                print(x)
            break
    return simp


def plot(x, y,  color, label, order, max_x, max_y, linewidth=1, markersize=12):
    markerfacecolor = color
    plt.subplot(3, 1, order)
    plt.plot(x, y, color=color, linewidth=linewidth, markerfacecolor=markerfacecolor, markersize=markersize,
             label=label)
    plt.grid()
    plt.legend([label])
    plt.ylim(0, max_y + 10)
    plt.xlim(0, (max_x + 2) / 30)
    plt.yticks(np.arange(0, max_y + 10, 5))
    plt.xticks(np.arange(0, max_x + 2, 200) / 30)
    plt.xlabel('Seconds')
    plt.ylabel('Fish')


df = pd.read_csv('Data1.csv')
Fish_Number_List = df['num'].tolist()

 
# Ordinary Average every 3 frames

maximum = []
time = []
frame = []
t = 0.0


MovAvgList = []
MovAvgFrameLst = []

for p in range(len(Fish_Number_List)):
    if (p+2) == len(Fish_Number_List):
        break
    MovAvgFrameLst.append(Fish_Number_List[p])
    MovAvgFrameLst.append(Fish_Number_List[p+1])
    MovAvgFrameLst.append(Fish_Number_List[p+2])
    MovAvgList.append(Average(MovAvgFrameLst))
    temp_lst = [Fish_Number_List[p], Fish_Number_List[p+1], Fish_Number_List[p+2]]

    maximum.append(max(temp_lst))


    frame.append(t)
    t += 1
    MovAvgFrameLst.clear()

frame.append(t + 0.1)
MovAvgList.append(Fish_Number_List[-2])
frame.append(t + 0.2)
MovAvgList.append(Fish_Number_List[-1])


for t in frame:
    time.append(t/30)

simp = simpler(Fish_Number_List)
simp2 = simpler(MovAvgList)
simp3 = simpler(maximum)

max_x = max([t])
max_y = max([max(MovAvgList), max(Fish_Number_List), max(maximum)])

plot(np.arange(0, len(simp), 1), simp, 'blue', 'Simplified Fish List', 1, max_x, max_y)
plot(np.arange(0, len(simp2), 1), simp2, 'red', 'Simplified Moving Average', 2,  max_x, max_y)
plot(np.arange(0, len(simp3), 1), simp3, 'orange', 'Simplified Moving Maximum', 3, max_x, max_y)

plt.show()

