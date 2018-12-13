import numpy as np


class TwoLayerNeuralNet:
    """
    Class of the two-layer neural network. The class initialize a model with random parameters and
    pass through the archetecture:
    a fully connected layer -> a Relu function -> a fully connected layer -> a softmax function.
    The loss and gradients are calculated using forward and back propagation, respectively.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Random initialization of the parameters of weights and biases.
        Arguments:
            input_size: the dimension D of the input data
            hidden_size: the number of neurons H in the hidden layer
            output_size: the number of classes C
        Output:
        A dictionary 'model' with the four parameters:
            model['W1']: first layer weights of shape (D, H)
            model['b1']: first layer biases of shape (H,)
            model['W2']: second layer weights of shape (H, C)
            model['b2']: second layer biases of shape (C,)
        """
        self.model = {}
        self.model['W1'] = 0.00001 * np.random.randn(input_size, hidden_size)   # small random values
        self.model['b1'] = np.zeros(hidden_size)                                # zeros
        self.model['W2'] = 0.00001 * np.random.randn(hidden_size, output_size)  # small random values
        self.model['b2'] = np.zeros(output_size)                                # zeros
        self.momentum_history = {}                                               # momentum history for SGD

    def nn(self, x, y=None, reg=0.0):
        """
        Define the structure of the two layer neural network. The loss and gradients are calculated using
        forward and back prop. The softmax loss function with L2 regularization is used. The hidden layer
        is passed through a Relu function to gain the non-linearity.
        Arguments:
            x: input training data of numpy.array(N, D)
            y: input training labels of shape numpy.array(N,) with label values lying in [0, C)
            reg: a float number of regularization strength
        Output:
            loss: softmax loss with L2 regularization from the forward prop
            grads: a dictionary of gradients for each parameters from the back prop
        """
        W1, W2, b1, b2 = self.model['W1'], self.model['W2'], self.model['b1'], self.model['b2']  # extract parameters
        N, D = x.shape

        h = np.dot(x, W1) + b1                                   # first fully connected layer
        h[h < 0] = 0                                             # Relu function
        y_hat = np.dot(h, W2) + b2                               # second fully connected layer
        y_hat = (y_hat.T - np.max(y_hat.T, axis=0)).T            # max trick
        p = (np.exp(y_hat).T / np.sum(np.exp(y_hat), axis=1)).T  # Softmax function

        # return predicted label if used for prediction
        if y is None:
            return np.argmax(y_hat, axis=1)

        # calculate loss
        LW = -1 / N * np.sum(np.log(p[range(N), y]))
        RW = 0.5 * (np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)))
        loss = LW + RW * reg

        # calculate gradients
        grads = {}
        p[range(len(y)), y] -= 1
        grads['b2'] = np.mean(p, axis=0)
        grads['W2'] = 1 / N * np.dot(h.T, p) + reg * W2
        delta = p.dot(W2.T) * (h > 0)  # from other sources, not quite sure how it works
        grads['b1'] = np.mean(delta, axis=0)
        grads['W1'] = 1 / N * np.dot(x.T, delta) + reg * W1

        return loss, grads

    def train(self, x, y, x_val, y_val, reg=0.0, lr=1e-6, momentum=0, lr_decay=0.95, decay_rate=0.95,
              method='momentum', mini_batch_SGD=True, num_epoch=50, batch_size=100):
        """
            Train the neural network with different methods.
            Arguments:
                x: input training data of numpy.array(N, D)
                y: input training labels of shape numpy.array(N,) with label values lying in [0, C)
                x_val: validation data of numpy.array(N_val, D)
                y_val: validation labels of shape numpy.array(N_val,) with label values lying in [0, C)
                reg: a float number of regularization strength
                lr: learning rate
                momentum: momentum in SGD with momentum
                decay_rate: learning rate decay used in RMS-prop
                lr_decay: learning rate decay used in training
                method: the update method to be used, can be one of 'sgd', 'momentum', or 'rmsprop'
                mini_batch_SGD: mini-batch SGD for parameter update if True, all data GD if False
                num_epoch: number of epoches
                batch_size: batch size used for mini-batch SGD
            Output:
                best_model: model with the best performance on the validation set
                loss_history: history of loss
                train_history: list of training accuracy for each epoch
                val_history: list of validation accuracy for each epoch
            """
        N = x.shape[0]  # number of data
        if mini_batch_SGD:
            iter_per_epoch = N // batch_size  # use mini-batch SGD
        else:
            iter_per_epoch = N  # use GD
        num_iters = num_epoch * iter_per_epoch  # number of iterations
        epoch = 0
        best_val_acc = 0.0
        best_model = {}
        loss_history = []
        train_history = []
        val_history = []

        # train
        for it in range(num_iters):
            # get batches of data
            if mini_batch_SGD:
                idx = np.random.choice(N, batch_size, replace=True)
                x_batch = x[idx]
                y_batch = y[idx]
            else:
                x_batch = x
                y_batch = y

            # calculate loss and gradients
            loss, grads = self.nn(x_batch, y_batch, reg)
            loss_history.append(loss)

            # update parameter
            for param in self.model:
                if method == 'sgd':
                    dx = - lr * grads[param]
                elif method == 'momentum':
                    if not param in self.momentum_history:
                        self.momentum_history[param] = np.zeros_like(grads[param])
                    self.momentum_history[param] = momentum * self.momentum_history[param] + lr * grads[param]
                    dx = - self.momentum_history[param]
                elif method == 'rmsprop':
                    if not param in self.momentum_history:
                        self.momentum_history[param] = np.zeros_like(grads[param])
                    self.momentum_history[param] = \
                        decay_rate * self.momentum_history[param] + (1 - decay_rate) * np.power(grads[param], 2)
                    dx = - lr / np.sqrt(1e-8 + self.momentum_history[param]) * grads[param]
                else:
                    raise ValueError('Method "%s" is not recognized or supported' % method)
                self.model[param] += dx

            # evaluate on the validation set
            if it == 0 or (it + 1) % iter_per_epoch == 0:
                if it != 0:
                    lr *= lr_decay  # decay the learning rate
                    epoch += 1

                # evaluate training accuracy with randomly chosen data
                if N >= 1000:
                    idx = np.random.choice(N, 1000, replace=True)
                    x_train_evl = x[idx]
                    y_train_evl = y[idx]
                else:
                    x_train_evl = x
                    y_train_evl = y
                y_pre = self.nn(x_train_evl)
                train_acc = np.mean(y_train_evl == y_pre)
                train_history.append(train_acc)

                # calculate evaluation set accuracy
                y_pre = self.nn(x_val)
                val_acc = np.mean(y_val == y_pre)
                val_history.append(val_acc)

                # update the best model so far
                if val_acc > best_val_acc:
                    best_model = self.model.copy()
                    best_val_acc = val_acc

                print('Finished epoch %d / %d: loss - %f, train accuracy - %f, validation accuracy - %f, '
                      'learning rate - %e' % (epoch, num_epoch, loss, train_acc, val_acc, lr))

        print('finished optimization. best validation accuracy: %f' % (best_val_acc,))

        return best_model, loss_history, train_history, val_history
