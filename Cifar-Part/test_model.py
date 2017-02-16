import time, pickle
import numpy as np
import theano
import theano.tensor as T
import cv2

num_epochs = 10
nb_classes = 101
data_dir = "/sharedfiles/food-101"
train_file = data_dir + "/meta/train.txt"
test_file  = data_dir + "/meta/test.txt"
batchsize  = 100
image_dim  = 320

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
        return a, list(a), label_to_id

train_images, labels, labels_to_id = load_food101_file(train_file)
test_images, _, _ = load_food101_file(test_file)

class Dataloader:
    def __init__(self,list_images,batch_size):
        self.list_images = list_images
        self.data_size = len(self.list_images)
        self.indices = np.arange(self.data_size)
        self.batch_size = batch_size
        self.nb_batches = self.data_size // self.batch_size
        np.random.shuffle(self.indices)
        pass

    def next_minibatch(self):
        np.random.shuffle(self.indices)
        inputs = np.zeros((self.batch_size,image_dim,image_dim,3))
        targets = np.zeros((self.batch_size,))
        for start_idx in range(0, self.data_size - self.batch_size + 1, self.batch_size):
            excerpt = self.indices[start_idx:start_idx + self.batch_size]
            print "loading next batch"
            for idx in range(self.batch_size):
                im = self.list_images[excerpt[idx]]
                try:
                    img = cv2.imread(data_dir + "/images/" + im + ".jpg")
                    img = cv2.resize(img,(image_dim,image_dim))
                except:
                    print data_dir + "/images/" + im + ".jpg"
                height, width, channel = img.shape
                inputs[idx] = img
                targets[idx] = labels_to_id[im.split("/")[0]]
            yield inputs, targets

X_train = np.asarray(train_set[0], dtype=theano.config.floatX).reshape((-1, 1, 28,28))
y_train = np.asarray(train_set[1], dtype='int32')

X_val = np.asarray(valid_set[0], dtype=theano.config.floatX).reshape((-1, 1, 28,28))
y_val = np.asarray(valid_set[1], dtype='int32')

X_test = np.asarray(test_set[0], dtype=theano.config.floatX).reshape((-1, 1, 28,28))
y_test = np.asarray(test_set[1], dtype='int32')

# image tensor
x = T.matrix()
# labels
y = T.ivector()

#### HERE TO FILL WITH MODEL
def model(input_var):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

print("Build model")
input_var = T.tensor4('inputs')
network = model(input_var)

print("Build loss function")
target_var = T.ivector('targets')
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
loss = loss.mean()


params = lasagne.layers.get_all_params(network, trainable=True)
updates = lasagne.updates.nesterov_momentum(
        loss, params, learning_rate=0.01, momentum=0.9)

test_prediction = lasagne.layers.get_output(network, deterministic=True)
test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                        target_var)
test_loss = test_loss.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                  dtype=theano.config.floatX)





def train_fn(inputs, targets):
	theano.function([input_var, target_var], loss, updates=updates)
    return inputs, targets
# to replace with compiled function train_fn = theano.function(..., inputs=[x,y], updates=updates)
# which returns  cost et accuracy


def val_fn(inputs, targets):
	theano.function([input_var, target_var], [test_loss, test_acc])
    return inputs, targets
# en realite, fonction compilee val_fn = theano.function(..., inputs=[x, y])

train_data_loader = Dataloader(train_images,batchsize)
test_data_loader = Dataloader(test_images,batchsize)

for epoch in range(num_epochs):
    train_err = 0
    start_time = time.time()
    for inputs, targets in train_data_loader.next_minibatch():
        print "training batch"
        print inputs.shape
        print targets.shape
        cost, acc = train_fn(inputs, targets)
        train_err += cost

    # Going over the validation data
    val_err = 0
    val_acc = 0
    for batch in test_data_loader.next_minibatch():
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        val_err += err
        val_acc += acc

 # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
    print("training loss:\t\t{:.6f}".format(train_err / train_data_loader.nb_batches))
    print("validation loss:\t\t{:.6f}".format(val_err / test_data_loader.nb_batches))
    print("validation accuracy:\t\t{:.2f} %".format(
                val_acc / test_data_loader.nb_batches * 100))
