"""Two Layer Network."""
# pylint: disable=invalid-name
import numpy as np
import matplotlib.pyplot as plt

# one hot encoder
def ohe(y, class_num):
    ohe = np.zeros((y.size,class_num), dtype = int)
    for idx, y_i in enumerate(y):
        ohe[idx, y_i] = 1
    return ohe
def sigmoid(x):
    x -= np.max(x)
    x = np.exp(x)
    x /= x.sum(axis = 1, keepdims = True)
    return x

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension
    of N, a hidden layer dimension of H, and performs classification over C
    classes. We train the network with a softmax loss function and L2
    regularization on the weight matrices. The network uses a ReLU nonlinearity
    after the first fully connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each
    class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.params['m1'] = np.zeros_like(self.params['W1'])
        self.params['v1'] = np.zeros_like(self.params['W1'])
        self.params['m2'] = np.zeros_like(self.params['W2'])
        self.params['v2'] = np.zeros_like(self.params['W2'])

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each
          y[i] is an integer in the range 0 <= y[i] < C. This parameter is
          optional; if it is not passed then we only return scores, and if it is
          passed then we instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c]
        is the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of
          training samples.
        - grads: Dictionary mapping parameter names to gradients of those
          parameters  with respect to the loss function; has the same keys as
          self.params.
        """
        # pylint: disable=too-many-locals
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, _ = X.shape
        if y is not None:
            num_classes = np.max(y)+1

        # Compute the forward pass
        scores = None
        ########################################################################
        # TODO: Perform the forward pass, computing the class scores for the   #
        # input. Store the result in the scores variable, which should be an   #
        # array of shape (N, C).                                               #
        ########################################################################
        z = X.dot(W1)+b1
        H = np.maximum(0, z)
        scores = H.dot(W2)+b2
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        loss = None
        ########################################################################
        # TODO: Finish the forward pass, and compute the loss. This should     #
        # include both the data loss and L2 regularization for W1 and W2. Store#
        # the result in the variable loss, which should be a scalar. Use the   #
        # Softmax classifier loss. So that your results match ours, multiply   #
        # the regularization loss by 0.5                                       #
        ########################################################################
        yHat = sigmoid(scores)

        loss = -np.sum(np.log(yHat[range(N), y]))
        loss /= N
        loss += 0.5*reg*(np.sum(W1**2) + np.sum(W2**2))
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        # Backward pass: compute gradients
        grads = {}
        ########################################################################
        # TODO: Compute the backward pass, computing the derivatives of the    #
        # weights and biases. Store the results in the grads dictionary. For   #
        # example, grads['W1'] should store the gradient on W1, and be a matrix#
        # of same size                                                         #
        ########################################################################
        # derivates/backprop
        dyHat = np.copy(yHat)
        dyHat[np.arange(N), y] -= 1
        dz = dyHat.dot(W2.T) * (z > 0)
        # gradients
        grads['W1'] = X.T.dot(dz)/N + reg*W1
        grads['b1'] = np.sum(dz, axis=0)/N
        grads['W2'] = H.T.dot(dyHat)/N + reg*W2
        grads['b2'] = np.sum(dyHat, axis=0)/N

        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False, method = 'SGD'):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means
          that X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning
          rate after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        # pylint: disable=too-many-arguments, too-many-locals
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            ####################################################################
            # TODO: Create a random minibatch of training data and labels,     #
            # storing hem in X_batch and y_batch respectively.                 #
            ####################################################################
            idx = np.random.choice(num_train, batch_size, replace = False)

            X_batch = X[idx]
            y_batch = y[idx]
            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            ####################################################################
            # TODO: Use the gradients in the grads dictionary to update the    #
            # parameters of the network (stored in the dictionary self.params) #
            # using stochastic gradient descent. You'll need to use the        #
            # gradients stored in the grads dictionary defined above.          #
            ####################################################################
            if method  == 'SGD':
                for i in ['W1', 'b1', 'W2', 'b2']:
                    self.params[i] -= learning_rate * grads[i]

            elif method == 'Adam':
                beta_mom = 0.9
                beta_vel = 0.95
                eta = 1e-8

                self.params['m1'] = beta_mom * self.params['m1'] + (grads['W1'])
                self.params['v1'] = beta_vel * self.params['v1'] + (1-beta_vel*grads['W1']**2)
                self.params['m2'] = beta_mom * self.params['m2'] + (grads['W2'])
                self.params['v2'] = beta_vel * self.params['v2'] + (1-beta_vel*grads['W2']**2)

                for i, j, k in zip(['W1', 'W2'], ['m1', 'm2'], ['v1', 'v2']):
                    self.params[i] -= learning_rate * (self.params[j]/(np.sqrt(self.params[k])+eta))
                self.params['b1'] -= learning_rate * grads['b1']
                self.params['b2'] -= learning_rate * grads['b2']

            else:
                raise NameError('Wrong optimization method: choose either SGD or Adam')

            ####################################################################
            #                             END OF YOUR CODE                     #
            ####################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each
          of the elements of X. For all i, y_pred[i] = c means that X[i] is
          predicted to have class c, where 0 <= c < C.
        """
        y_pred = None

        ########################################################################
        # TODO: Implement this function; it should be VERY simple!             #
        ########################################################################
        z = X.dot(self.params['W1'])+self.params['b1']
        H = np.maximum(0, z)
        scores = H.dot(self.params['W2'])+self.params['b2']
        y_ohe = sigmoid(scores)
        y_pred = np.argmax(y_ohe, axis = 1)
        ########################################################################
        #                              END OF YOUR CODE                        #
        ########################################################################

        return y_pred


def neuralnetwork_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    best_net = None # store the best model into this

    ############################################################################
    # TODO: Tune hyperparameters using the validation set. Store your best     #
    # trained model in best_net.                                               #
    #                                                                          #
    # To help debug your network, it may help to use visualizations similar to #
    # the  ones we used above; these visualizations will have significant      #
    # qualitative differences from the ones we saw above for the poorly tuned  #
    # network.                                                                 #
    #                                                                          #
    # Tweaking hyperparameters by hand can be fun, but you might find it useful#
    # to  write code to sweep through possible combinations of hyperparameters #
    # automatically like we did on the previous exercises.                     #
    ############################################################################
    # visualization function, copied from jupyter notebook
    def debug_helper(stats, params):
        # show lr, reg... in plot
        textstr = '\n'.join((
                        r'learning rate: %e'%params['lr'],
                        r'regularization: %e'%params['reg'],
                        r'hidden layer size: %e'%params['hidden'],
                        r'no. of epochs: %e'%params['epochs']))
        props = dict(boxstyle='round', alpha=0.5)

        # Plot the loss function and train / validation accuracies
        fig, ax = plt.subplots(nrows=2, ncols=1)

        ax1 = plt.subplot(2, 1, 1)
        plt.plot(stats['loss_history'])
        plt.title('Loss history')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        ax2 = plt.subplot(2, 1, 2)
        plt.plot(stats['train_acc_history'], label='train')
        plt.plot(stats['val_acc_history'], label='val')
        plt.title('Classification accuracy history')
        plt.xlabel('Epoch')
        plt.ylabel('Clasification accuracy')

        plt.tight_layout()

        ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)

        plt.show()

    def refine_params(params, best_params):
        lrs = sorted(params['lr'])
        regs = sorted(params['reg'])
        h_sizes = sorted(params['hidden'])
        epochs = sorted(params['epochs'])
        print(lrs, regs, h_sizes, epochs)
        print(best_params)

        new_range = {}
        for param, string in zip([lrs, regs, h_sizes, epochs],
                                            ['lr', 'reg', 'hidden', 'epochs']):
            best_val = best_params[string]
            best_idx = np.argmax(param ==  best_val)
            if best_idx != 0 and best_idx != len(params[string])-1:
                new_range[string] =  [param[best_idx - 1], param[best_idx + 1]]
            elif best_idx == 0:
                new_low = best_val - (param[1] - best_val)
                print('best_val: %e, higher value: %e'%(best_val, param[best_idx + 1]),'\n','new low: %e' %new_low)
                new_range[string] =  [new_low, param[best_idx + 1]]
            else:
                new_high =  best_val + (best_val - param[-2])
                print('best_val: %e, lower value: %e'%(best_val, param[best_idx - 1]),'\n','new high: %e'%new_high)
                print(best_idx)
                new_range[string] = [param[-2], new_high]

        print(new_range)
        return new_range

    def update_params(params, best_params, tuning_size):
        new_ranges = refine_params(params, best_params)

        # lr_range = np.arange(new_ranges['lr'][0], new_ranges['lr'][1], 1e-8)
        # reg_range = np.arange(new_ranges['reg'][0], new_ranges['reg'][1], 1e2)
        # hidden_range = np.arange(new_ranges['hidden'][0], new_ranges['hidden'][1], 10)
        # epoch_range =  np.arange(new_ranges['epochs'][0], new_ranges['epochs'][1], 1)

        for param, step_size in zip(['lr', 'reg', 'hidden'],
                                        [1e-8, 1e2, 10, 1]):
        #, 'epochs'
            try:
                params[param] = np.random.choice(np.arange(new_ranges[param][0], new_ranges[param][1], step_size), tuning_size, replace = False)
            except ValueError:
                params[param] = np.random.choice(np.arange(new_ranges[param][0], new_ranges[param][1], step_size/100), tuning_size, replace = False)

        return params

    def accuracy(pred, label):
        return np.mean((pred == label),)

    # function to do perform grid search and refine range every round
    def optimize(params, X_train, y_train, X_val, y_val, num_iters):
        # initialize variables
        lrs = params['lr']
        regs = params['reg']
        h_sizes = params['hidden']
        #epochs = params['epochs']

        best_lr = 0
        best_reg = 0
        best_h_size = 0
        best_epoch = 0

        num_train, dim = X_train.shape

        stats = None
        best_model = None
        best_val = -1

        # loop over parameters and train with each combination
        for lr in lrs:
            for reg in regs:
                for h_size in h_sizes:
                    #for epoch in epochs:
                    print('learning_rate: %e'%lr,
                            'regularization: %e'%reg,
                            'hidden_size: %i'%h_size,
                            #'no. epochs: %i'%epoch
                            )
                    batch_size = 250
                    no_epochs = 10
                    num_iters = int(X_train.shape[0]/batch_size*no_epochs)
                    print('iterations: %i'%num_iters)
                    model = TwoLayerNet(input_size = dim, hidden_size = h_size,
                                        output_size =  10)
                    stats = model.train(X_train, y_train, X_val, y_val,
                               learning_rate = lr,
                               reg = reg, num_iters = num_iters,
                               batch_size = batch_size)
                    train_acc = accuracy(model.predict(X_train), y_train)
                    val_acc = accuracy(model.predict(X_val), y_val)
                    if val_acc > best_val:
                        best_stats = stats
                        best_val = val_acc
                        best_model = model
                        best_lr = lr
                        best_reg = reg
                        best_h_size = h_size
                        #best_epoch = epoch
        best_params = {
                        'lr': best_lr,
                        'reg':  best_reg,
                        'hidden': best_h_size,
                        #'epochs': best_epoch,
                        }

        return (best_model, best_params, best_stats, best_val)

    # initialize all variables
    stats_history = {}
    stats_history['loss_history'] = []
    stats_history['train_acc_history'] = []
    stats_history['val_acc_history'] = []

    lr_range = np.arange(1.0e-7, 5.1e-7, 1e-8)
    reg_range = np.arange(1, 1e2, 1)
    hidden_range = np.arange(300, 1500, 10)
    epoch_range =  np.arange(3, 10, 1)

    params = {}
    params['lr'] = []
    params['reg'] = []
    params['hidden'] = []
    #params['epochs'] = []

    tuning_size = 5
    tuning_rounds = 5

    for param_range, param in zip([lr_range, reg_range, hidden_range, epoch_range],
                        ['lr', 'reg', 'hidden']):
    # , 'epochs'
        params[param] = np.random.choice(param_range, tuning_size, replace = False)

    # hyperparameter optimization
    def hyper_opt(X_train, y_train, X_val, y_val, params, tuning_size,
                    tuning_rounds, num_iters):
        round_count = 1
        best_val = -1
        best_model = None
        best_params = None
        new_model = None
        stats_list = []
        while round_count <= tuning_rounds:
            print('Starting round no. %i'%round_count)
            new_model, new_params, stats, new_val = \
                        optimize(params, X_train, y_train, X_val, y_val, num_iters)
            print('This rounds best model achieved: %f' %new_val)
            stats_list.append(stats)
            if new_val > best_val:
                print('New best model found!')
                best_val = new_val
                best_model = new_model
                best_params = new_params
                debug_helper(stats, best_params)
                params = update_params(params, best_params, tuning_size)
            round_count += 1
        return best_val, best_model, best_params, stats_list

    #best_val, best_model, best_params, stats_list = \
    #            hyper_opt(X_train, y_train, X_val, y_val, params,
    #                        tuning_size, tuning_rounds, 1)
    best_val = -1
    for lr in [0.5e-3, 1e-3]:
        #for reg in [0.1]:
        model = TwoLayerNet(X_train.shape[1], 150, 10)
        stats  = model.train(X_train, y_train, X_val, y_val,
            num_iters=3500, batch_size=200,
            learning_rate=lr, learning_rate_decay=0.95,
            reg=0.1, verbose=True)
        val_acc = (model.predict(X_val) == y_val).mean()
        if val_acc > best_val:
            best_val = val_acc
            best_net = model

            plt.subplots(nrows=2, ncols=1)

            plt.subplot(2, 1, 1)
            plt.plot(stats['loss_history'])
            plt.title('Loss history')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')

            plt.subplot(2, 1, 2)
            plt.plot(stats['train_acc_history'], label='train')
            plt.plot(stats['val_acc_history'], label='val')
            plt.title('Classification accuracy history')
            plt.xlabel('Epoch')
            plt.ylabel('Clasification accuracy')

            plt.tight_layout()
            plt.show()

            print(val_acc)

    best_net = best_model

    ############################################################################
    #                               END OF YOUR CODE                           #
    ############################################################################
    return best_net
