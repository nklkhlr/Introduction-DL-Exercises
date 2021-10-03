"""Linear Softmax Classifier."""
# pylint: disable=invalid-name
import numpy as np

from .linear_classifier import LinearClassifier

from multiprocessing import Pool

def opt(X_train, y_train, X_val, y_val, lr, reg, results, all_classifiers, best_softmax,
        best_val, bset_lr, best_reg):
    print('start')
    sm = SoftmaxClassifier()
    # train model
    sm.train(X_train, y_train, learning_rate = lr, reg = reg,
             num_iters = 1, batch_size = 250)
    # predict training and validation and calculate accuracy
    train_acc = accuracy(sm.predict(X_train), y_train)
    val_acc = accuracy(sm.predict(X_val), y_val)
    # store in results
    results[(lr, reg)] = (train_acc, val_acc)
    all_classifiers.append(sm)
    if val_acc > best_val:
        best_softmax = sm
        best_val = val_acc
        best_lr = lr
        best_reg = reg

    return results, all_classifiers, best_softmax, best_val, bset_lr, best_reg

def cross_entropoy_loss_naive(W, X, y, reg):
    """
    Cross-entropy loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # pylint: disable=too-many-locals
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient using explicit     #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    def s(x,W):
        xHat = x.dot(W)
        # for numerical stability
        xHat -= np.max(xHat)
        xHat = np.exp(xHat)
        x_sum = np.sum(xHat)
        return xHat/x_sum

    def ohe(y):
        ohe = np.zeros((y.size+1, 10))
        for idx, y_i in enumerate(y):
            ohe[idx, y_i] = 1
        return ohe

    def loss_func(yHat, y):
        return np.multiply(y, np.log(yHat)+np.multiply((1-y), np.log(1-yHat)))

    y_ohe = ohe(y)
    for idx_i, (x_i, y_i) in enumerate(zip(X,y_ohe)):
        yHat_i = s(x_i, W)
        # sum to loss
        loss += loss_func(yHat_i, y_i)
        for idx_j, x_ij in enumerate(x_i):
            for idx_k, y_ik in enumerate(y_i):
                # add to gradient matrix
                dW[idx_j, idx_k] = (yHat_i[idx_k]-y_i[idx_k])*X[idx_i, idx_j]

    # normalize to sample number
    loss = np.sum(loss)
    loss /= y.size
    dW /= y.size
    # add regularization
    loss += reg*np.sum(W**2)
    dW += reg*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


def cross_entropoy_loss_vectorized(W, X, y, reg):
    """
    Cross-entropy loss function, vectorized version.

    Inputs and outputs are the same as in cross_entropoy_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the cross-entropy loss and its gradient without explicit   #
    # loops. Store the loss in loss and the gradient in dW. If you are not     #
    # careful here, it is easy to run into numeric instability. Don't forget   #
    # the regularization!                                                      #
    ############################################################################
    def ohe(y):
        ohe = np.zeros((np.size(y), 10))
        for idx, y_i in enumerate(y):
            ohe[idx, y_i] = 1
        return ohe
    y_ohe = ohe(y)

    yHat = X.dot(W)
    # ensure numeric stability
    yHat -= np.max(yHat)
    yHat = np.exp(yHat)
    # softmax
    yHat /= yHat.sum(axis=1, keepdims=True)

    #loss += np.sum(y_ohe*np.log(yHat)+(1-y_ohe)*np.log(1-yHat))
    loss -= np.sum(np.log(yHat[range(y.shape[0]), y]))
    loss /= y.size
    loss += reg*np.sum(W**2)

    dW = X.T.dot(yHat-y_ohe)
    dW /= y.size
    dW += reg*W
    ############################################################################
    #                          END OF YOUR CODE                                #
    ############################################################################

    return loss, dW


class SoftmaxClassifier(LinearClassifier):
    """The softmax classifier which uses the cross-entropy loss."""

    def loss(self, X_batch, y_batch, reg):
        return cross_entropoy_loss_vectorized(self.W, X_batch, y_batch, reg)


def softmax_hyperparameter_tuning(X_train, y_train, X_val, y_val):
    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the
    # fraction of data points that are correctly classified.
    results = {}
    best_val = -1
    best_softmax = None
    all_classifiers = []
    learning_rates = [1e-7, 5e-7]
    regularization_strengths = [2.5e4, 5e4]

    ############################################################################
    # TODO:                                                                    #
    # Write code that chooses the best hyperparameters by tuning on the        #
    # validation set. For each combination of hyperparameters, train a         #
    # classifier on the training set, compute its accuracy on the training and #
    # validation sets, and  store these numbers in the results dictionary.     #
    # In addition, store the best validation accuracy in best_val and the      #
    # Softmax object that achieves this accuracy in best_softmax.              #                                      #
    #                                                                          #
    # Hint: You should use a small value for num_iters as you develop your     #
    # validation code so that the classifiers don't take much time to train;   #
    # once you are confident that your validation code works, you should rerun #
    # the validation code with a larger value for num_iters.                   #
    ############################################################################

    # Optimization using GridSearch
    def accuracy(pred, label):
        return np.mean((pred == label),)

    lrs = np.random.choice(np.arange(1.0e-7, 5.1e-7, 1e-8), 15, replace = False)
    regs = np.random.choice(np.arange(1e4, 2e4, 1e3), 15, replace = False)
    best_lr = 0
    best_reg = 0
    i = 1
    j = 1
    for lr in lrs:
        j = 1
        sm = SoftmaxClassifier()
        for reg in regs:
            print(i,j)
            # train model
            sm.train(X_train, y_train, learning_rate = lr, reg = reg,
                     num_iters = 1000, batch_size = 250)
            # predict training and validation and calculate accuracy
            train_acc = accuracy(sm.predict(X_train), y_train)
            val_acc = accuracy(sm.predict(X_val), y_val)
            # store results
            results[(lr, reg)] = (train_acc, val_acc)
            all_classifiers.append(sm)
            if val_acc > best_val:
                best_softmax = sm
                best_val = val_acc

            j += 1
        if best_val > 38:
                break
        i += 1

    # using parallelization
    # args = [(X_train, y_train, X_val, y_val, lr, reg, results, all_classifiers, best_softmax,
    #     best_val, best_lr, best_reg) for lr in lrs for reg in regs]
    # mp = Pool()
    # results, all_classifiers, best_softmax, best_val, bset_lr, best_reg = mp.starmap(opt, args)
    # mp.close()
    # mp.join()
    ############################################################################
    #                              END OF YOUR CODE                            #
    ############################################################################

    # Print out results.
    for (lr, reg) in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
              lr, reg, train_accuracy, val_accuracy))

    print('best validation accuracy achieved during validation: %f' % best_val)
    return best_softmax, results, all_classifiers
