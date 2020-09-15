import mnist_loader
train, val, test = mnist_loader.load_data_wrapper()

import network
net = network.Network([784, 30, 10])
# this will take some time to train depends on your machine 
net.SGD(train, 30, 10, 3, test_data = test)

