import numpy as np
from hw2 import preprocessing as P
import torch
import torch.utils.data
from hw2 import all_cnn as A

trainX = np.load('../dataset/train_feats.npy')
print(np.shape(trainX))
train_label = np.load('../dataset/train_labels.npy')
testX = np.load('../dataset/test_feats.npy')
#train_fea, test_fea = P.cifar_10_preprocess(trainX, testX)
#np.save('train_fea.npy', train_fea)
#np.save('test_fea.npy', test_fea)
train_fea = np.load('train_fea.npy')
test_fea = np.load('test_fea.npy')
'''
trainX_subset = trainX[:100,]
testX_subset = testX[:100, ]
train_label_subset = train_label[:100]
np.save('trainX_subset.npy', trainX_subset)
np.save('train_label_subset.npy', train_label_subset)
np.save('testX_subset.npy', testX_subset)

trainX = np.load('trainX_subset.npy')
testX = np.load('testX_subset.npy')
train_fea, test_fea = P.cifar_10_preprocess(trainX, testX)
train_label = np.load('train_label_subset.npy')
# train_fea: (100, 3, 32, 32)
# test_fea: (100, 3, 32, 32)
'''

print('Begin initialization.')


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


trainY_tensor = to_tensor(train_label)
trainY_tensor = trainY_tensor.type('torch.LongTensor')


class TestDataSet(torch.utils.data.TensorDataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return np.shape(self.data_tensor)[0]


print('Finish initialization.')

LEARN_RATE = 0.001
BATCH_SIZE = 128
def training_routine(data, labels_binary, num_epochs, minibatch_size, learn_rate):
    my_net = A.all_cnn_module()  # Create the network,
    loss_fn = torch.nn.CrossEntropyLoss()  # and choose the loss function / optimizer
    optim = torch.optim.SGD(my_net.parameters(), lr=learn_rate, momentum=0.9)
    if torch.cuda.is_available():
        my_net = my_net.cuda()
        loss_fn = loss_fn.cuda()

    dataset = torch.utils.data.TensorDataset(to_tensor(data), labels_binary)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=minibatch_size, shuffle=True)

    for epoch in range(num_epochs):
        losses = []
        i = 0
        for (input_val, label) in data_loader:
            optim.zero_grad()  # Reset the gradients
            prediction = my_net(to_variable(input_val))  # Feed forward # [torch.FloatTensor of size 11x10]
            loss = loss_fn(prediction, to_variable(label))  # Compute losses
            loss.backward()  # Backpropagate the gradients
            losses.append(loss.data.cpu().numpy())
            optim.step()  # Update the network
        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))
    return my_net


trained_net = training_routine(train_fea, trainY_tensor, 2, 128, 0.001)
test_data = TestDataSet(to_tensor(train_fea))
test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False)
test_data_real = TestDataSet(to_tensor(test_fea))
test_real_loader = torch.utils.data.DataLoader(
    test_data_real, batch_size=1, shuffle=False)


def classification(net, loader):
    label_lst = []
    for input_val in loader:
        input = to_variable(input_val)
        y = net(input)
        label = np.argsort(y.data.cpu().numpy())[0][-1]
        label_lst.append(label)
    return label_lst


def accuracy(label_lst, truth):
    n = len(label_lst)
    acc = 0
    for i in range(n):
        if label_lst[i] == int(truth[i]):
            acc += 1
    return acc, acc / n


if __name__ == '__main__':
    trained_label = classification(trained_net, test_loader)
    acc = accuracy(trained_label, train_label)
    test_label = classification(trained_net, test_real_loader)
    np.save('test_label.npy', test_label)
    print(acc)
    for i in range(100):
        print(trained_label[i], train_label[i])