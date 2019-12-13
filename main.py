import numpy as np
import Network as NWRK
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import sys

fl = input("model file:")
if(fl != ""):
    md = NWRK.Network()
    md.load(fl)
else:
    md = NWRK.Network([784,64,32,10], activ=["sigmoid", "sigmoid", "softmax"], loss="mse", learning_rate=5e-2)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

data = np.vstack([img.reshape(-1) for img in mnist.train.images])
data = data.reshape(data.shape + (1,))
labels = np.zeros((55000,10))
labels[np.arange(55000), mnist.train.labels] = 1

X_test = np.vstack([img.reshape(-1,) for img in mnist.test.images])
X_test = X_test.reshape(X_test.shape + (1,))
y_test = mnist.test.labels

for epoch in tqdm(range(100)): # number of epochs
    random_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(random_state)
    np.random.shuffle(labels)

    for dp in tqdm(range(550)):
        md.update(data[dp*100:(dp+1)*100], labels[dp*100:(dp+1)*100])
    
    # Now test.
    correct = 0
    test_attempted = 0
    total_cost = 0
    for i in range(200):
        test_attempted += 1
        raw_pred = md.predict(X_test[i])
        oh_label = np.zeros(10)
        oh_label[y_test[i]] = 1
        total_cost += NWRK.Cost.mse(oh_label, raw_pred)
        pred = np.argmax(raw_pred)
        if(pred == y_test[i]):
            correct += 1
    print("\n\n", str(correct/test_attempted), str(total_cost/test_attempted), "\n\n")

    md.save("./model.json")