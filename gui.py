__author__ = 'jason'
from Tkinter import *
from PIL import Image, ImageTk
import h5py
import numpy as np
import json
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

def image_getter(filename):
    data = h5py.File(filename)
    def get_image(i):
        im = Image.fromarray(np.uint8(data['images'][i]*255.0))
        return im
    num_images = len(data['images'])
    shape = (data['images'][0].shape[0], data['images'][0].shape[1])
    return get_image, num_images, shape

class DataViewer(object):
    def __init__(self, image_getter, num_images, im_size, rows, cols):
        self.master = Tk()
        self.labels = []
        self.im_size = im_size
        self._page = 0
        self.image_getter = image_getter
        self.num_images = num_images
        self.page_size = (rows, cols)
        self.frame = Frame(self.master, width=im_size[0]*cols, height=im_size[1]*rows)
        self.frame.bind('<Right>', self.next_page)
        self.frame.bind('<Left>', self.prev_page)
        self.frame.grid(row=1,column=0, columnspan=cols, rowspan=rows-1)
        self.results_frame = Frame(self.master)
        self.results_frame.grid(row=0, column=cols, columnspan=4, rowspan=rows)

        for row in range(self.page_size[0]):
            for column in range(self.page_size[1]):
                im  = self.image_getter(row*column)
                im = im.resize((im_size[0]*2, im_size[1]*2), Image.ANTIALIAS)
                im = ImageTk.PhotoImage(im)
                button = Button(self.frame, image=im)
                button.im = im
                button.grid(row=row, column=column)
                self.labels.append(button)
        self.frame.focus_set()

    def plot(self, results):
        import io
        print("Hello")
        weights = h5py.File(os.path.join(results, 'weights.hdf5'))
        plt.figure(figsize=(12,12))
        filters = weights['layer_0/param_0'][:]
        plt.figure(figsize=(8,8))
        print(filters.shape)
        filters = filters - filters.min()
        filters = filters / filters.max()
        print(filters[0])
        for i in range(filters.shape[0]):
            plt.subplot(1, filters.shape[0], i+1)
            plt.imshow(np.mean(np.transpose(filters[i], (1, 2, 0)), axis=2), interpolation='bilinear', cmap=plt.cm.gray)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True)
        # with open(os.path.join(results, 'history.json'), 'r') as hist:
        #     history = json.load(hist)
        #     plt.figure(figsize=(4,4))
        #     plt.plot(history['epoch'], history['history']['acc'])
        #     buf = io.BytesIO()
        #
        #     plt.savefig(buf, format='png', transparent=True)
        buf.seek(0)
        im = Image.open(buf)
        im = ImageTk.PhotoImage(im)
        button = Label(self.results_frame, image=im)
        button.im = im
        button.grid(row=0, column=0, columnspan=16)

    def next_page(self, event):
        self.page(1)

    def prev_page(self, event):
        self.page(-1)

    def page(self, offset):
        self._page += offset
        increment = self.page_size[0]*self.page_size[1]
        if self._page < 0:
            self._page = 0
        if self._page > self.num_images/increment - 1:
            self._page = self.num_images/increment - 1

        ctr = 0
        offset = self._page*(self.page_size[0]*self.page_size[1])
        for row in range(self.page_size[0]):
            for column in range(self.page_size[1]):
                im  = self.image_getter(row*column + offset)
                im = im.resize((self.im_size[0]*2, self.im_size[1]*2), Image.ANTIALIAS)
                im = ImageTk.PhotoImage(im)

                button = self.labels[ctr]
                button.configure(image=im)
                button.im = im
                ctr += 1

    def run(self):
         mainloop()

if __name__ == "__main__":
    g = DataViewer(*image_getter('./data/letters.hdf5'), rows=8, cols=16)
    g.run()
