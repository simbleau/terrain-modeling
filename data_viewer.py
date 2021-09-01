#!/usr/bin/python

import sys
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("Please enter a file, i.e. `python view.py file.tiff`")
    sys.exit(1)

path = sys.argv[1]
x = plt.imread(path)
plt.imshow(x)
plt.colorbar()
plt.show()
