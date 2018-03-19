
# coding: utf-8

# In[1]:


import numpy as np
import torch
from torch import nn
from collections import Counter
TEST = 0
BATCH_SIZE = 80
EMBEDDING_DIM = 400
NUM_EPOCHES = 1
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 1150
LOAD_PRE = False
SAVE_RESULTS = True
prepath = '../params/params.pt'
savepath = '../params/param1.pt'
VALIDATION = True
PREDICTION = False
BATCH_LENGTH = 70
GENERATION = False
SMOOTHING = 0.1


# In[2]:


vocab = np.load('../dataset/vocab.npy')
wiki_train = np.load('../dataset/wiki.train.npy')
wiki_valid = np.load('../dataset/wiki.valid.npy')
if TEST:
    wiki_train = wiki_train[:10]
    wiki_valid = wiki_valid[:10]
train_array = np.concatenate(wiki_train)
valid_array = np.concatenate(wiki_valid)
charcount = vocab.shape[0]
if not LOAD_PRE:
    word_count = dict(Counter(list(train_array)))
    count_list = np.zeros(charcount)
    for i in word_count.keys():
        count_list[i] = word_count[i]
    count_list = count_list/charcount
    count_smooth = count_list * (1 - SMOOTHING) + SMOOTHING/charcount
    bias_unigram = np.log(count_smooth)

# In[3]:


def loader(word_array):
    trainX = word_array[:-1]
    trainY = word_array[1:]
    data_loader = []
    # (N//batch size, batch size)
    trainX_batch = np.reshape(trainX[:BATCH_SIZE * (trainX.shape[0] // BATCH_SIZE)],
                              (BATCH_SIZE, trainX.shape[0] // BATCH_SIZE)).T
    trainY_batch = np.reshape(trainY[:BATCH_SIZE * (trainY.shape[0] // BATCH_SIZE)],
                              (BATCH_SIZE, trainY.shape[0] // BATCH_SIZE)).T
    L = trainX_batch.shape[0] // BATCH_LENGTH + 1
    for i in range(L):
        minibatchX = to_tensor(trainX_batch[BATCH_LENGTH * i: BATCH_LENGTH * (i + 1)])
        minibatchY = to_tensor(trainY_batch[BATCH_LENGTH * i: BATCH_LENGTH * (i + 1)])
        minibatchX = to_variable(minibatchX.type('torch.LongTensor'))
        minibatchY = to_variable(minibatchY.type('torch.LongTensor'))
        data_loader.append((minibatchX, minibatchY))
    return data_loader


# In[16]:


def test_loader(test_batch):
    L = test_batch.shape[0]
    data_loader = []
    for i in range(L):
        minibatchX = to_variable(to_tensor(test_batch[i,:]).type('torch.LongTensor'))
        data_loader.append(minibatchX)
    return data_loader


# In[5]:


def to_tensor(numpy_array):
    # Numpy array -> Tensor
    return torch.from_numpy(numpy_array).float()


def to_variable(tensor):
    # Tensor -> Variable (on GPU if possible)
    if torch.cuda.is_available():
        # Tensor -> GPU Tensor
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)


# In[6]:


train_loader = loader(train_array)
vali_loader = loader(valid_array)
# generate test array
test_batch = np.reshape(valid_array[: 10 * 300], (10, 300))


# In[88]:


class WikiModel(nn.Module):
    def __init__(self, charcount, embedding_dim, hidden_size):
        super(WikiModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=charcount, embedding_dim=embedding_dim)
        self.rnns = nn.ModuleList([
            nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True),
            nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True),
            nn.LSTM(input_size=hidden_size, hidden_size=embedding_dim, batch_first=True),
        ])
        self.projection = nn.Linear(in_features=embedding_dim, out_features=charcount)

        if not LOAD_PRE:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.bias = nn.Parameter(to_tensor(bias_unigram))
            

    def forward(self, input, forward=0):
        h = input  # (n, t)
        h = self.embedding(h)
        states = []
        for rnn in self.rnns:
            h, state = rnn(h)
            states.append(state)
        h = self.projection(h)
        logits = h
        if forward > 0:
            outputs = []
            h = torch.max(logits, dim=2)[1] # 300*10*33278 -- 300 * 10
            for i in range(forward):
                h1 = self.embedding(h)
                for j, rnn in enumerate(self.rnns):
                    h1, state = rnn(h1, states[j])
                    states[j] = state
                h = self.projection(h1) # 300*10*33278
                h = torch.max(h, dim=2)[1] # 300 * 10
                outputs.append(h[-1,:].data)
            logits = torch.stack(outputs)
        return logits


# In[8]:


def training_routine(data_loader, num_epochs, learn_rate,
                     charcount, embedding_dim, hidden_size):
    my_net = WikiModel(charcount, embedding_dim, hidden_size)  # Create the network
    if LOAD_PRE:
        my_net.load_state_dict(torch.load(prepath))
    loss_fn = torch.nn.CrossEntropyLoss()  # and choose the loss function / optimizer
    optim = torch.optim.Adam(my_net.parameters(), lr=learn_rate, weight_decay=0.001)

    if torch.cuda.is_available():
        my_net = my_net.cuda()
        loss_fn = loss_fn.cuda()

    for epoch in range(num_epochs):
        losses = []
        i = 0
        for (input_val, label) in data_loader:
            optim.zero_grad()  # Reset the gradients
            prediction = my_net(input_val)  # Feed forward
            prediction = prediction.view(-1, prediction.size()[2])
            label = label.view(-1)
            loss = loss_fn(prediction, label)  # Compute losses
            loss.backward()  # Backpropagate the gradients
            losses.append(loss.data.cpu().numpy())
            optim.step()  # Update the network
            print("Epoch {} Batch {} Loss: {:.4f}".format(epoch, i, loss.data[0]))
            i += 1
        print("Epoch {} Loss: {:.4f}".format(epoch, np.asscalar(np.mean(losses))))
    return my_net


# In[9]:


def validation(vali_loader):
    # make prediction for one batch of articles
    vali_net = WikiModel(charcount, EMBEDDING_DIM, HIDDEN_SIZE)
    vali_net.load_state_dict(torch.load(savepath))
    vali_net.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = []

    if torch.cuda.is_available():
        vali_net = vali_net.cuda()
        loss_fn = loss_fn.cuda()

    for (input_val, label) in vali_loader:
        prediction = vali_net(input_val)
        prediction = prediction.view(-1, prediction.size()[2])
        label = label.view(-1)
        loss = loss_fn(prediction, label)
        losses.append(loss.data.cpu().numpy())
    print("Loss on validation set: {:.4f}".format(np.asscalar(np.mean(losses))))
    return


# In[10]:


def prediction(inp):
    # prediction on a new array batch.
    inp_var = to_variable(to_tensor(inp.T).type('torch.LongTensor'))
    pre_net = WikiModel(charcount, EMBEDDING_DIM, HIDDEN_SIZE)
    pre_net.load_state_dict(torch.load(savepath))
    output_full = pre_net(inp_var)
    output = output_full[-1,:, :]
    return output


# In[85]:


def generate(inp, forward):
    # generate a new array of given length.
    inp_var = to_variable(to_tensor(inp.T).type('torch.LongTensor'))
    if torch.cuda.is_available():
        inp_var = inp_var.cuda()
    pre_net = WikiModel(charcount, EMBEDDING_DIM, HIDDEN_SIZE)
    pre_net.load_state_dict(torch.load(savepath))
    pre_net.eval()
    classes = pre_net(inp_var, forward=forward)
    return classes


def generate_text(outputs):
    # generate text from given indexes
    outputs = torch.t(outputs)
    text = vocab[outputs]
    return text


# In[12]:


my_net = training_routine(train_loader, NUM_EPOCHES, LEARNING_RATE, charcount, EMBEDDING_DIM, HIDDEN_SIZE)
print('Saving model parameters and making predictions...')
if SAVE_RESULTS:
    torch.save(my_net.state_dict(), savepath)


# In[18]:


if VALIDATION:
    validation(vali_loader)
    print('Finish validation.')


# In[17]:


if PREDICTION:
    pre = prediction(test_batch)
    print('Finish prediction.')


# In[89]:


if GENERATION:
    gene_array = generate(test_batch, GENERATION)
    print('Finish generation.')
    print(generate_text(gene_array))
