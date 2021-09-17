#!/usr/bin/python
from time import sleep

import numpy

from helper_methods import *
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plotwithargs2D(path):
    x = plt.imread(path)
    plt.imshow(x)
    plt.colorbar()
    plt.show()

def plotwithargs3D(pathin):
    path = 'terrain/' + pathin
    model = 'output/' + pathin + '.h5'
    model = load_model(model)
    fig = plt.figure(figsize=(100, 100))
    ax = fig.add_subplot(111, projection='3d')
    x, y = get_xy(path)
    y_hat = model.predict(x)
    y_hat[np.isnan(y_hat)] = 0
    xlist = []
    ylist = []
    zlist = []
    counter = 0
    for i in range(-180, 180):
        for j in range(-180, 180):
            zlist.append(y[counter])
            ylist.append(j)
            xlist.append(i)
            counter += 1
    xlist = numpy.array(xlist)
    ylist = numpy.array(ylist)
    zlist = numpy.array(zlist)
    xlist = xlist.flatten()
    ylist = ylist.flatten()
    zlist = zlist.flatten()
    y_hat = y_hat.flatten()
    ax.plot(xlist, ylist, zlist, '.', color='red', markersize=1)
    ax.plot(xlist, ylist, y_hat, '.', color='blue', markersize=1)
    #plt.show()
    plt.savefig('output/test' + '.png', dpi = 10)





if len(sys.argv) == 2:
    plotwithargs2D(sys.argv[1])
else:
    plotwithargs3D('Grand_Canyon_0.1deg.tiff')
