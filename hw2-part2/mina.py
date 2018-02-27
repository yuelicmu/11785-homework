import torch
import numpy as np
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn

#trainX = np.load('sample_data/train_feas.npy')
#train_label = np.load('sample_data/train_lab.npy')

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
def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor, require_grad=False):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor, requires_grad=require_grad)


class UtteranceData(torch.utils.data.TensorDataset):
    def __init__(self, data_matrix, label):
        self.data_matrix = data_matrix
        self.label = label

    def __getitem__(self, index):
        features = self.data_matrix[index, 0].transpose()
        phoneme_seg = self.data_matrix[index, 1]
        assignment = np.zeros(np.shape(features)[1], dtype=np.int32)
        n = np.shape(phoneme_seg)[0]
        for i in range(n - 1):
            assignment[phoneme_seg[i]:phoneme_seg[i + 1]] = (i + 1)
        assignment[phoneme_seg[n - 1]:] = n
        labels = self.label[index]
        lab_mask = np.ones(np.shape(labels), dtype=np.int32)
        return features, assignment, labels, lab_mask

    def __len__(self):
        return np.shape(self.data_matrix)[0]


def create_batch(sample):
    N = np.shape(sample)[0]
    M = np.max([np.shape(sample[i][0])[1] for i in range(N)])
    padded_features = [np.pad(sample[i][0], ((0, 0), (0, M - np.shape(sample[i][0])[1])),
                              'constant', constant_values=0) for i in range(N)]
    features_batch = torch.stack([torch.from_numpy(b) for b in padded_features], 0)
    max_time = np.max([np.shape(sample[i][1])[0] for i in range(N)])
    padded_assign = [np.pad(sample[i][1], (0, max_time - np.shape(sample[i][1])[0]),
                            'constant', constant_values=0) for i in range(N)]
    assignment_batch = torch.stack([torch.from_numpy(d) for d in padded_assign], 0).type('torch.LongTensor')
    max_phon = np.max([np.shape(sample[i][2])[0] for i in range(N)])
    padded_label = [np.pad(sample[i][2], (0, max_phon - np.shape(sample[i][2])[0]),
                           'constant', constant_values=0) for i in range(N)]
    labels_batch = torch.stack([torch.from_numpy(d) for d in padded_label], 0).type('torch.LongTensor')
    padded_mask = [np.pad(sample[i][3], (0, max_phon - np.shape(sample[i][3])[0]),
                          'constant', constant_values=0) for i in range(N)]
    mask_batch = torch.stack([torch.from_numpy(d) for d in padded_mask], 0).type('torch.LongTensor')
    return features_batch, assignment_batch, labels_batch, mask_batch


class UtteranceTestData(torch.utils.data.TensorDataset):
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix

    def __getitem__(self, index):
        features = self.data_matrix[index, 0].transpose()
        phoneme_seg = self.data_matrix[index, 1]
        assignment = np.zeros(np.shape(features)[1], dtype=np.int32)
        n = np.shape(phoneme_seg)[0]
        for i in range(n - 1):
            assignment[phoneme_seg[i]:phoneme_seg[i + 1]] = (i + 1)
        assignment[phoneme_seg[n - 1]:] = n
        return features, assignment

    def __len__(self):
        return np.shape(self.data_matrix)[0]


def create_batch_test(sample):
    N = np.shape(sample)[0]
    M = np.max([np.shape(sample[i][0])[1] for i in range(N)])
    padded_features = [np.pad(sample[i][0], ((0, 0), (0, M - np.shape(sample[i][0])[1])),
                              'constant', constant_values=0) for i in range(N)]
    features_batch = torch.stack([torch.from_numpy(b) for b in padded_features], 0)
    max_time = np.max([np.shape(sample[i][1])[0] for i in range(N)])
    padded_assign = [np.pad(sample[i][1], (0, max_time - np.shape(sample[i][1])[0]),
                            'constant', constant_values=0) for i in range(N)]
    assignment_batch = torch.stack([torch.from_numpy(d) for d in padded_assign], 0).type('torch.LongTensor')
    return features_batch, assignment_batch


def CNNa():
    seq = nn.Sequential(
        nn.Conv1d(40, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(512, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv1d(192, 46, kernel_size=3, stride=1, padding=1),
    )
    return seq


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"


class _Loss(nn.Module):
    def __init__(self, size_average=True):
        super(_Loss, self).__init__()
        self.size_average = size_average


class NewLoss(_Loss):
    def __init__(self, assignment, lab_mask, size_average=False):
        super(NewLoss, self).__init__(size_average)
        self.assignment = assignment
        self.lab_mask = lab_mask
        max_phone = self.assignment.max()
        n, p = self.assignment.size()
        Assign = torch.zeros(n, max_phone + 1, p)
        for i in range(n):
            for j in range(p):
                Assign[i, self.assignment[i, j], j] = 1
        Assign = Assign[:, 1:, :]
        self.Assign = Assign
        hot_sum = Assign.sum(2)
        self.hot_sum = hot_sum

    def forward(self, input, target):
        lab_mask = self.lab_mask.type('torch.ByteTensor')
        output = to_variable(torch.masked_select(target, lab_mask))
        _assert_no_grad(output)
        prediction_t = torch.transpose(input, 1, 2)
        assign_variable = to_variable(self.Assign, require_grad=False)
        phone_out = torch.bmm(assign_variable, prediction_t)
        n, p = self.assignment.size()
        hot_sum = torch.reciprocal(torch.max(self.hot_sum, torch.ones(self.hot_sum.size())))
        hot_sum = hot_sum.view(n, self.assignment.max(), 1)
        hot_sum_3d = to_variable(hot_sum.expand(phone_out.size()), require_grad=False)
        phone_out = phone_out * hot_sum_3d
        phone_view_nonzero = phone_out.view(-1, 46)[lab_mask.view(-1).nonzero(), :].view(-1, 46)
        return F.cross_entropy(phone_view_nonzero, output, size_average=self.size_average)


# Training
def training_routine(data_loader, num_epochs, learn_rate, param):
    my_net = CNNa()
    # optim = torch.optim.Adam(my_net.parameters(), lr=learn_rate, weight_decay=0.001)
    optim = torch.optim.Adam(param, lr=learn_rate, weight_decay=0.001)

    if torch.cuda.is_available():
        my_net = my_net.cuda()

    for epoch in range(num_epochs):
        losses = []
        for (features, assignment, labels, lab_mask) in data_loader:
            loss_fn = NewLoss(assignment, lab_mask)
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()
            optim.zero_grad()  # Reset the gradients
            prediction = my_net(to_variable(features))  # Feed forward
            # torch.FloatTensor, n * 46 * max(time)
            loss = loss_fn(prediction, labels)  # calculating loss
            loss.backward()  # Backpropagate the gradients
            losses.append(loss.data.cpu().numpy())
            optim.step()  # Update the network
        #print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.sum(losses))), end=',')
        print("Loss: {:.4f}".format(np.asscalar(np.sum(losses))), end=',')
    return my_net


# the output of network is
def predict_single_batch(net, loader):
    label_lst = []
    for (features, assignment) in loader:
        input = to_variable(features)
        y = torch.squeeze(net(input)).t()
        assign = assignment.view(-1)
        max_phone = assign.max()
        p = assignment.size()[1]
        Assign = torch.zeros(max_phone + 1, p)
        for j in range(p):
            Assign[assign[j], j] = 1
        Assign = Assign[1:, :]
        hot_sum = Assign.sum(1).view(-1, 1)
        assign_variable = to_variable(Assign, require_grad=False)
        phone_out = torch.mm(assign_variable, y)
        hot_sum = torch.reciprocal(torch.max(hot_sum, torch.ones(hot_sum.size())))
        hot_sum = to_variable(hot_sum, require_grad=False)
        phone_out = phone_out * hot_sum
        # phone_out: N * 46 * max(phone)
        label = np.array(np.argsort(phone_out.data.cpu().numpy())[:, -1], dtype=np.int32)
        label_lst.append(label)
    return label_lst


def accuracy(label_lst, truth):
    n = len(label_lst)
    acc = 0
    N = 0
    for i in range(n):
        acc += np.nonzero(label_lst[i]-truth[i])[0].shape[0]
        N += label_lst[i].shape[0]
    return acc, acc / N


print('Begin initialization:')
BATCH_SIZE = 10
data_set = UtteranceData(trainX, train_label)
data_loader = torch.utils.data.DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True,
                                          collate_fn=create_batch)
data_set_dev = UtteranceTestData(devX)
data_loader_dev = torch.utils.data.DataLoader(data_set_dev, batch_size=1, shuffle=False,
                                              collate_fn=create_batch)
data_set_test = UtteranceTestData(testX)
data_loader_test = torch.utils.data.DataLoader(data_set_test, batch_size=1, shuffle=False,
                                               collate_fn=create_batch_test)
print('Finish initialization.')

my_net = CNNa()
my_net.apply(weights_init)
params = my_net.parameters()
for i in range(20):
    print("Epoch {}".format(i), end=' ')
    my_net = training_routine(data_loader, 1, 0.0001, params)
    params = my_net.parameters()
    f = 'net_params/epoch'+str(i)+'.pt'
    torch.save(my_net.state_dict(), f)
    label = predict_single_batch(my_net, data_loader_dev)
    acc = accuracy(label, dev_label)
    print(" accuracy {} %.".format(round(acc[1]*100, 2)))
