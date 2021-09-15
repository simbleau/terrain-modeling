#!/usr/bin/python

from helper_methods import *
import matplotlib.pyplot as plt
import numpy as np


def plot(files):
    output_folder = 'output/'
    input_folder = 'terrain/'
    for file in files:
        print(f"Working on file: {file}")
        input_path = input_folder + file
        output_path = output_folder + file + '.h5'

        # Get x and y
        x, y = get_xy(input_path)

        rows = []
        columns = []
        for x1 in range(360):
            row = []
            for x2 in range(360):
                row.append(y[x1 * 360 + x2][0])
            rows.append(row)
        for x1 in range(360):
            column = []
            for x2 in range(360):
                column.append(y[x1 + x2 * 360][0])
            columns.append(column)

        avg_height_per_row = []
        avg_height_per_column = []
        for row in rows:
            avg_height_per_row.append(np.average(row))
        for column in columns:
            avg_height_per_column.append(np.average(column))

        fig, axs = plt.subplots(2, sharex=True, sharey=True)
        fig.suptitle(file)
        axs[0].plot(avg_height_per_row)
        axs[0].title.set_text('Average height per row')
        axs[1].plot(avg_height_per_column)
        axs[1].title.set_text('Average height per column')
        plt.show()


def plot_all():
    all_files = ['Appalachian_State_0.1deg.tiff',
                 'Appalachian_State_1.0deg.tiff',
                 'Appalachian_State_2.0deg.tiff',
                 'Grand_Canyon_0.1deg.tiff',
                 'Grand_Canyon_1.0deg.tiff',
                 'Grand_Canyon_2.0deg.tiff',
                 'NC_Coast_1.0deg.tiff',
                 'NC_Coast_2.0deg.tiff',
                 'NC_Coast_3.0deg.tiff']
    plot(all_files)


plot_all()
