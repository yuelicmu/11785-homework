from wsj_loader import WSJ
import numpy as np

# loading training data
loader = WSJ()
trainX, trainY = loader.train
N = np.shape(trainX)[0]
m = sum([np.shape(trainX[i])[0] for i in range(N)])
#print(m) # 15449191
trainX_pad = np.empty((m, 40))
trainY_pad = np.empty(m)
train_utter_len = np.empty(m)
train_utter_idx = np.empty(m)
line_idx = 0
for n in range(N):
    l = np.shape(trainX[n])[0]
    trainX_pad[line_idx:(line_idx+l), ] = trainX[n]
    trainY_pad[line_idx:(line_idx+l)] = trainY[n]
    train_utter_len[line_idx:(line_idx+l)] = l * np.ones(l)
    train_utter_idx[line_idx:(line_idx+l)] = np.array(range(l))
    line_idx += l

np.save('processed/trainX_pad', trainX_pad)
np.save('processed/trainY_pad', trainY_pad)
np.save('processed/train_utter_len', train_utter_len)
np.save('processed/train_utter_idx', train_utter_idx)

# loading validation data
devX, devY = loader.dev
N = np.shape(devX)[0]
m = sum([np.shape(devX[i])[0] for i in range(N)])
# print(m) # 669294
devX_pad = np.empty((m, 40))
devY_pad = np.empty(m)
dev_utter_idx = np.empty(m)
dev_utter_len = np.empty(m)
line_idx = 0
for n in range(N):
    l = np.shape(devX[n])[0]
    devX_pad[line_idx:(line_idx+l), ] = devX[n]
    devY_pad[line_idx:(line_idx+l)] = devY[n]
    dev_utter_idx[line_idx:(line_idx+l)] = np.array(range(l))
    dev_utter_len[line_idx:(line_idx+l)] = l * np.ones(l)
    line_idx += l
np.save('processed/devX_pad', devX_pad)
np.save('processed/devY_pad', devY_pad)
np.save('processed/dev_utter_len', dev_utter_len)
np.save('processed/dev_utter_idx', dev_utter_idx)

# loading training data set
# testX = loader.test
testX = np.load('data/test.npy')
N = np.shape(testX)[0]
# print(N) # 268
m = sum([np.shape(testX[i])[0] for i in range(N)])
# print(m) # 169656
# print(min([np.shape(testX[i])[0] for i in range(N)])) # 177
testX_pad = np.empty((m, 40))
test_utter_idx = np.empty(m)
test_utter_len = np.empty(m)
line_idx = 0
for n in range(N):
    l = np.shape(testX[n])[0]
    testX_pad[line_idx:(line_idx+l), ] = testX[n]
    test_utter_idx[line_idx:(line_idx+l)] = np.array(range(l))
    test_utter_len[line_idx:(line_idx+l)] = l * np.ones(l)
    line_idx += l
np.save('processed/testX_pad', testX_pad)
np.save('processed/test_utter_len', test_utter_len)
np.save('processed/test_utter_idx', test_utter_idx)
