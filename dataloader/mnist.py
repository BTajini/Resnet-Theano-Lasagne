import gzip, pickle
import numpy as np
import theano

data_dir = "/sharedfiles/"

class Dataloader:
    def __init__(self,batch_size,split="train", shuffle=True):
        print("Loading data " + split)
        with gzip.open(data_dir + "mnist.pkl.gz", 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f)
        if split == "train" :
            self.inputs = np.asarray(train_set[0], dtype=theano.config.floatX).reshape((-1, 1, 28,28))
            self.targets = np.asarray(train_set[1], dtype='int32')
        elif split == "val" :
            self.inputs = np.asarray(valid_set[0], dtype=theano.config.floatX).reshape((-1, 1, 28,28))
            self.targets = np.asarray(valid_set[1], dtype='int32')
        else:
            self.inputs = np.asarray(test_set[0], dtype=theano.config.floatX).reshape((-1, 1, 28,28))
            self.targets = np.asarray(test_set[1], dtype='int32')

        self.data_size = self.inputs.shape[0]
        self.indices = np.arange(self.data_size)
        self.batch_size = batch_size
        self.nb_batches = self.data_size // self.batch_size
        self.shuffle = shuffle

    def shape(self):
        return self.inputs.shape

    def next_minibatch(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start_idx in range(0, self.data_size - self.batch_size + 1, self.batch_size):
            excerpt = self.indices[start_idx:start_idx + self.batch_size]
            yield self.inputs[excerpt], self.targets[excerpt]
