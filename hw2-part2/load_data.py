import numpy as np
# loading data
trainX = np.load('data/train-features.npy')
# print(np.shape(trainX)) #(24590, 2)
# np.shape(trainX[0,0]) (477, 40) np.shape(trainX[0,1]) (57,)
train_label = np.load('data/train-labels.npy')
# np.shape(train_label) # (24590,)
# np.shape(train_label[0]) # (57,)
devX = np.load('data/dev-features.npy')
# np.shape(devX) # (1103, 2)
dev_label = np.load('data/dev-labels.npy')
# np.shape(dev_label) # (1103,)
testX = np.load('data/test-features.npy')
# np.shape(testX) # (268, 2)
