import gzip, pickle
import numpy as np
import lasagne

data_dir = "/sharedfiles/"

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict

class Dataloader:
    def __init__(self,batch_size,split="train", shuffle=True):
        print("Loading data " + split)
        xs = []
        ys = []
        if split == "train":
            for j in range(5):
                d = unpickle(data_dir+'cifar-10-batches-py/data_batch_'+`j+1`)
                xs.append(d['data'])
                ys.append(d['labels'])
        else:
            d = unpickle(data_dir+'cifar-10-batches-py/test_batch')
            xs.append(d['data'])
            ys.append(d['labels'])

        x = np.concatenate(xs)/np.float32(255)
        y = np.concatenate(ys)
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

        # subtract per-pixel mean
        pixel_mean = np.mean(x[0:50000],axis=0)
        #pickle.dump(pixel_mean, open("cifar10-pixel_mean.pkl","wb"))
        x -= pixel_mean

        # create mirrored images
        x_flip = x[:,:,:,::-1]
        y_flip = y
        x = np.concatenate((x,x_flip),axis=0)
        y = np.concatenate((y,y_flip),axis=0)

        self.inputs=lasagne.utils.floatX(x)
        self.targets=y.astype('int32')
        self.data_size = self.inputs.shape[0]
        self.indices = np.arange(self.data_size)
        self.batch_size = batch_size
        self.nb_batches = self.data_size // self.batch_size
        self.shuffle = shuffle

    def shape(self):
        return self.inputs.shape

    def next_minibatch(self, augment=False):
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start_idx in range(0, self.data_size - self.batch_size + 1, self.batch_size):
            excerpt = self.indices[start_idx:start_idx + self.batch_size]
            if augment:
                # as in paper :
                # pad feature arrays with 4 pixels on each side
                # and do random cropping of 32x32
                padded = np.pad(self.inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
                random_cropped = np.zeros(self.inputs[excerpt].shape, dtype=np.float32)
                crops = np.random.random_integers(0,high=8,size=(self.batch_size,2))
                for r in range(self.batch_size):
                    random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
                inp_exc = random_cropped
            else:
                inp_exc = self.inputs[excerpt]
            yield inp_exc, self.targets[excerpt]
