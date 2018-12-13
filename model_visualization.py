from load_data import load_data
from neural_net import TwoLayerNeuralNet
import numpy as np
import matplotlib.pyplot as plt


def predict(x, model):
    W1, W2, b1, b2 = model['W1'], model['W2'], model['b1'], model['b2']
    h = np.dot(x, W1) + b1
    h[h < 0] = 0
    y_hat = np.dot(h, W2) + b2
    y_hat = np.argmax(y_hat, axis=1)
    return y_hat

def main():
    x_train, y_train, x_test, y_test = load_data('data/cifar-10-batches-py')  # load dataset

    # create training/validation/test sets
    N_train, N_test = x_train.shape[0], x_test.shape[0]
    percent_val, percent_train = 0.02, 0.1   # define percentage of validation and test data to be used
    # reshape data to rows
    x_train = x_train.reshape(N_train, -1)
    x_test = x_test.reshape(N_test, -1)
    # since the dataset is already shuffled, the train/val/test sets can be defined with simply slicing
    x_val = x_train[int(-N_train*percent_val):]
    y_val = y_train[int(-N_train*percent_val):]
    x_train = x_train[:int(-N_train*percent_val)]
    y_train = y_train[:int(-N_train*percent_val)]
    x_test = x_test[:int(N_test*percent_train)]
    y_test = y_test[:int(N_test*percent_train)]

    # normalize the image data
    image_mean = np.mean(x_train, axis=0)
    x_train -= image_mean
    x_val -= image_mean
    x_test -= image_mean

    print('Training data shape: ', x_train.shape, '     Train labels shape: ', y_train.shape)
    print('Validation data shape: ', x_val.shape, '     Train labels shape: ', y_val.shape)
    print('Test data shape:     ', x_test.shape, '     Test labels shape: ', y_test.shape)
    print()

    classifier = TwoLayerNeuralNet(32*32*3, 1000, 10)  # initialize the neural net
    best_model, loss_history, train_history, val_history = \
        classifier.train(x_train, y_train, x_val, y_val, reg=0.0001, lr=1e-4, momentum=0.9, lr_decay=0.7,
                         decay_rate=0.95, method='momentum', mini_batch_SGD=True, num_epoch=20, batch_size=100)

    # plot the loss, accuracy curves
    plt.subplot(2, 1, 1)
    plt.plot(loss_history)
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(2, 1, 2)
    plt.plot(train_history)
    plt.plot(val_history)
    plt.legend(['Training accuracy', 'Validation accuracy'], loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Clasification accuracy')
    plt.show()

    test_pre = predict(x_test, best_model)
    print('Test accuracy: ', np.mean(test_pre == y_test))


if __name__ == "__main__":
    main()
