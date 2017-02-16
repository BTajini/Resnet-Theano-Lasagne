import theano
import numpy as np
import cv2

min_size = 256
crop_size = 224

def load_food101_file(filename):
    '''Load a single file of food101'''
    with open(filename, 'rb') as f:
        a = []
        labels = set()
        for line in f:
            a.append(line.rstrip())
            labels.add(line.rstrip().split("/")[0])
        label_to_id = {}
        i = 0
        for l in labels:
            label_to_id[l] = i
            i = i + 1
        return a, list(labels), label_to_id


def scale(im, size):
    h, w, channel = im.shape
    if (w <= h and w == size) or (h <= w and h == size):
        return im
    if w <= h:
        return cv2.resize(im, (size, int(h * 1.0 / w * size) ))
    else:
        return cv2.resize(im, ( int(w * 1.0 / h * size), size ))

import math
def centerCrop(im, size):
    w1 = math.ceil((im.shape[1] - size) / 2.0)
    h1 = math.ceil((im.shape[0] - size) / 2.0)
    return im[ h1 : size + h1 , w1 : size + w1]


class Dataloader:
    def __init__(self,batch_size,split="train",shuffle=True):
        if split=="train":
            self.image_paths, self.labels, self.labels_to_id = load_food101_file("food-101/meta/train.txt")
        else:
            self.image_paths, self.labels, self.labels_to_id = load_food101_file("food-101/meta/test.txt")

        self.data_size = len(self.image_paths)
        self.indices = np.arange(self.data_size)
        self.batch_size = batch_size
        self.nb_batches = self.data_size // self.batch_size
        self.shuffle = shuffle

    def shape(self):
        return (self.data_size,3,crop_size,crop_size)

    def next_minibatch(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        inputs = np.zeros((self.batch_size,3,crop_size,crop_size), dtype=theano.config.floatX)
        targets = np.zeros((self.batch_size,), dtype='int32')
        for start_idx in range(0, self.data_size - self.batch_size + 1, self.batch_size):
            excerpt = self.indices[start_idx:start_idx + self.batch_size]
            for idx in range(self.batch_size):
                im = self.image_paths[excerpt[idx]]
                try:
                    img = cv2.imread("food-101/images/" + im + ".jpg")
                    img = scale(img, min_size)
                    img = centerCrop(img, crop_size)
                except Exception as e:
                    print e
                    print "food-101/images/" + im + ".jpg"
                inputs[idx,0] = img[...,0]
                inputs[idx,1] = img[...,1]
                inputs[idx,2] = img[...,2]
                targets[idx] = self.labels_to_id[im.split("/")[0]]
            yield inputs, targets
