from matplotlib import pyplot
from keras.datasets import cifar10

import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import metrics, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt


def load_data(filename):
    data = pd.read_csv(filename)

    # set Date feature to date time object and sort data by date
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.sort_values('Date')

    return data


def data_visualization(data):
    # data visualization
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title('Closing Price')
    ax1.plot(data['Date'], data['Close'])

    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title('Volume')
    ax2.plot(data['Date'], data['Volume'])

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def train_test_split(data, train_percentage):
    # Split train and test data
    length = data.shape[0]
    train_size = round(length * train_percentage)
    train_data = data[0:train_size]
    test_data = data[train_size:length]

    return train_data, test_data


def main():
    # load data
    data = load_data('VTI.csv')

    # visualize data
    #data_visualization(data)

    # Split training and testing data
    train_data, test_data = train_test_split(data, 0.7)











if __name__ == '__main__':
    main()