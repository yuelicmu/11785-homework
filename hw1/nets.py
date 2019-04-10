import torch
class MLPa(torch.nn.Module):
    """
    Network Structure:
    Input - n*2n,relu - 2n*2n, relu - 2n*n, sigmoid - n*out - CrossEntropy
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        count_bitwidth = 138
        self.to_hidden1 = torch.nn.Linear(input_size, 2 * input_size)
        self.hidden_relu1 = torch.nn.ReLU()
        self.to_hidden2 = torch.nn.Linear(2 * input_size, 2 * input_size)
        self.hidden_relu2 = torch.nn.ReLU()
        self.to_hidden3 = torch.nn.Linear(2 * input_size, input_size)
        self.hidden_sigmoid3 = torch.nn.Sigmoid()
        self.to_binary = torch.nn.Linear(input_size, count_bitwidth)

    def forward(self, input_val):
        hidden1 = self.hidden_relu1(self.to_hidden1(input_val))
        hidden2 = self.hidden_relu2(self.to_hidden2(hidden1))
        hidden3 = self.hidden_sigmoid3(self.to_hidden3(hidden2))
        return self.to_binary(hidden3)


class MLPb(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        count_bitwidth = 138
        self.to_hidden1 = torch.nn.Linear(input_size, int(input_size / 2))
        self.hidden_relu1 = torch.nn.ReLU()
        self.to_hidden2 = torch.nn.Linear(int(input_size / 2), int(input_size / 3))
        self.hidden_relu2 = torch.nn.ReLU()
        self.to_hidden3 = torch.nn.Linear(int(input_size / 3), 300)
        self.hidden_relu3 = torch.nn.ReLU()
        self.to_hidden4 = torch.nn.Linear(300, 200)
        self.hidden_relu4 = torch.nn.ReLU()
        self.to_binary = torch.nn.Linear(200, count_bitwidth)

    def forward(self, input_val):
        hidden1 = self.hidden_relu1(self.to_hidden1(input_val))
        hidden2 = self.hidden_relu2(self.to_hidden2(hidden1))
        hidden3 = self.hidden_relu3(self.to_hidden3(hidden2))
        hidden4 = self.hidden_relu4(self.to_hidden4(hidden3))
        return self.to_binary(hidden4)
