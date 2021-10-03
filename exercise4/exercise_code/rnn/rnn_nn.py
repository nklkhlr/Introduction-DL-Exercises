import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                          nonlinearity = activation)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        h_seq, h = self.rnn(x)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc  = nn.Linear(hidden_size, 10)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq=[]
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        _, batch_size, hidden_size = x.shape

        # initialize hidden and cell state vector if none given
        if h is None:
            h = torch.zeros(1, batch_size, hidden_size)
        if c is None:
            c = torch.zeros(1, batch_size, hidden_size)

        h_seq, (h, c) = self.lstm(x, (h, c))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , (h, c)


class RNN_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, activation="relu" ):
        super(RNN_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a RNN classifier                                            #
        ############################################################################
        self.rnn = nn.RNN(input_size = input_size, hidden_size = hidden_size,
                          nonlinearity = activation)
        self.fc = nn.Linear(hidden_size, classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        x, h = self.rnn(x)
        x = self.fc(x)
        print(x.shape)

        return x
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

class LSTM_Classifier(torch.nn.Module):
    def __init__(self,classes=10, input_size=28 , hidden_size=128, num_layers = 1):
        super(LSTM_Classifier, self).__init__()
        ############################################################################
        #  TODO: Build a LSTM classifier                                           #
        ############################################################################
        self.classes = classes
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers)
        self.fc  = nn.Linear(hidden_size, self.classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def forward(self, x):
        ############################################################################
        #                                YOUR CODE                                 #
        ############################################################################
        # initialize hidden and cell state vector
        batch_size = x.size(1)

        h_init = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c_init = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        h_seq, (h, c) = self.lstm(x, (h_init, c_init))
        #print(h_seq.shape)
        out = self.fc(h_seq[-1,:,:])
        #print(out.shape)
        return out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

