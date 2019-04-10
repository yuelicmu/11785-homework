import numpy as np
import torch
import torch.utils.data
from nets import MLPa, MLPb

SUBSET_TRAIN = 0

# load data, construct data set
trainX = np.load('processed/trainX_pad.npy')
trainY = np.load('processed/trainY_pad.npy')
train_utter_idx = np.load('processed/train_utter_idx.npy')
train_utter_len = np.load('processed/train_utter_len.npy')
devX = np.load('processed/devX_pad.npy')
devY = np.load('processed/devY_pad.npy')
dev_utter_idx = np.load('processed/dev_utter_idx.npy')
dev_utter_len = np.load('processed/dev_utter_len.npy')
testX = np.load('processed/testX_pad.npy')
test_utter_idx = np.load('processed/test_utter_idx.npy')
test_utter_len = np.load('processed/test_utter_len.npy')
NUM_NEIGHBOR = 10
BATCH_SIZE = 128

if SUBSET_TRAIN == 1:
    m1 = 59840
    trainX = trainX[:m1, :]
    trainY = trainY[:m1]
    train_utter_idx = train_utter_idx[:m1]
    train_utter_len = train_utter_len[:m1]
    devX = devX[:9071, :]
    devY = devY[:9071]
    dev_utter_idx = dev_utter_idx[:9071]
    dev_utter_len = dev_utter_len[:9071]


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


trainY_tensor = to_tensor(trainY)
trainY_tensor = trainY_tensor.type('torch.LongTensor')


# construct data set
class ContextDataSet(torch.utils.data.TensorDataset):

    def __init__(self, data_tensor, target_tensor, utter_length, utter_idx, num_neighbor):
        super().__init__(data_tensor, target_tensor)
        self.utter_length = utter_length
        self.utter_idx = utter_idx
        self.num_neighbor = num_neighbor

    def __getitem__(self, index):
        utter_idx = self.utter_idx[index]
        utter_len = self.utter_length[index]
        utter_idx_new = min(max(utter_idx, self.num_neighbor), utter_len - 1 - self.num_neighbor)
        index_new = index + utter_idx_new - utter_idx
        idx_lst = list(range(int(index_new - self.num_neighbor), int(index_new + self.num_neighbor + 1)))
        idx_lst.remove(index)
        idx_lst.append(index)
        return self.data_tensor[idx_lst, :].view(-1), self.target_tensor[index]

    def __len__(self):
        return np.shape(self.data_tensor)[0]


# construct data set for training
class TestDataSet(torch.utils.data.TensorDataset):

    def __init__(self, data_tensor, utter_length, utter_idx, num_neighbor):
        self.data_tensor = data_tensor
        self.utter_length = utter_length
        self.utter_idx = utter_idx
        self.num_neighbor = num_neighbor

    def __getitem__(self, index):
        utter_idx = self.utter_idx[index]
        utter_len = self.utter_length[index]
        utter_idx_new = min(max(utter_idx, self.num_neighbor), utter_len - 1 - self.num_neighbor)
        index_new = index + utter_idx_new - utter_idx
        idx_lst = list(range(int(index_new - self.num_neighbor), int(index_new + self.num_neighbor + 1)))
        idx_lst.remove(index)
        idx_lst.append(index)
        return self.data_tensor[idx_lst, :].view(-1)

    def __len__(self):
        return np.shape(self.data_tensor)[0]


print('Begin initialization.')
m1 = 59840
data_set = ContextDataSet(to_tensor(trainX), trainY_tensor, train_utter_len, train_utter_idx, NUM_NEIGHBOR)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)
data_set_dev = TestDataSet(to_tensor(devX), dev_utter_len, dev_utter_idx, NUM_NEIGHBOR)
data_loader_dev = torch.utils.data.DataLoader(data_set_dev, batch_size=1, shuffle=False)
data_set_test = TestDataSet(to_tensor(testX), test_utter_len, test_utter_idx, NUM_NEIGHBOR)
data_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=1, shuffle=False)
data_set_subset = TestDataSet(to_tensor(trainX[:m1, :]), train_utter_len[:m1], train_utter_idx[:m1], NUM_NEIGHBOR)
data_loader_subset = torch.utils.data.DataLoader(data_set_subset, batch_size=1, shuffle=False)
print('Finish initialization.')


# training_procedure
def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


def training_routine(data_loader, num_epochs, learn_rate):
    my_net = MLPb((2 * NUM_NEIGHBOR + 1) * 40)  # Create the network,
    loss_fn = torch.nn.CrossEntropyLoss()  # and choose the loss function / optimizer
    optim = torch.optim.SGD(my_net.parameters(), lr=learn_rate, momentum=0.9)

    for epoch in range(num_epochs):
        losses = []
        for (input_val, label) in data_loader:
            optim.zero_grad()  # Reset the gradients
            prediction = my_net(to_variable(input_val))  # Feed forward
            loss = loss_fn(prediction, to_variable(label))  # Compute losses
            loss.backward()  # Backpropagate the gradients
            losses.append(loss.data.cpu().numpy())
            optim.step()  # Update the network
        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))
    return my_net


# output results
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
    my_net = training_routine(data_loader, 10, 0.001)
    label_dev = classification(my_net, data_loader_dev)
    label_test = classification(my_net, data_loader_test)
    np.save('label_dev.npy', label_dev)
    [acc, acc_rate] = accuracy(label_dev, devY)
    label_train = classification(my_net, data_loader_subset)
    [acc_train, acc_rate_train] = accuracy(label_train, trainY[:59840])
    print('Training data:')
    print(acc_train, acc_rate_train)
    print('Test data:')
    print(acc, acc_rate)
    np.save('label_test.npy', label_test)
    