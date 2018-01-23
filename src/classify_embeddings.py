from sklearn import svm
from sklearn.manifold import TSNE

import pickle

from data_utils import *
from utils import plot_tsne

import ConvLSTMAE

def tsne(embedded_X_train, y, f_name, perplexity=10):
    # tsne
    for perpex in [7, 20, 50, 100]:
        X_embedded = TSNE(n_components=2, perplexity=perpex).fit_transform(embedded_X_train)  # perplexity
        plot_tsne(X_embedded, y, f_name + '_' + str(perpex))


def fit_conv_ae(model, out_dir):
    svm_data = MultipleDatasets(directory="./UCR_TS_Archive_2015",
                                datasets=test_datasets, merge_train_test=False, data_length=512, val_percentage=0.2)
    svm_data.load_data()

    best_result = 0
    better_than_timenet = 0
    for dataset_name in test_datasets:
        data = svm_data.get_dataset(dataset_name)
        embedded_X_train = model.embeddings(data['X_train'])

        tsne(embedded_X_train, data['Y_train'], out_dir + '/' + dataset_name)

        # fit svm       C, gamma, rbf
        clf = svm.SVC()
        clf.fit(embedded_X_train, data['Y_train'])

        # save
        output = open(out_dir + '/' + dataset_name + '.pkl', 'wb')
        pickle.dump(clf, output)

        y_val = data['Y_val']
        X_val = model.embeddings(data['X_val'])

        y_pred = clf.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        print 1 - accuracy <= Best_Results[dataset_name], 1 - accuracy <= Timenet_Results[
            dataset_name], dataset_name, 1 - accuracy, Timenet_Results[dataset_name], Best_Results[dataset_name]


def fit_lstm(args, model, out_dir):
    svm_data = MultipleDatasets(directory="./UCR_TS_Archive_2015",
                                datasets=test_datasets, merge_train_test=False, val_percentage=0.2)
    svm_data.load_data()

    best_result = 0
    better_than_timenet = 0
    for dataset_name in test_datasets:
        data = svm_data.get_dataset(dataset_name)
        embedded_X_train = ConvLSTMAE.embeddings(args, svm_data, data['X_train'], model)

        print 'svm getting', embedded_X_train.shape
        tsne(embedded_X_train, data['Y_train'], out_dir + '/' + dataset_name)

        # fit svm       C, gamma, rbf
        clf = svm.SVC(C=1.0)
        clf.fit(embedded_X_train, data['Y_train'])

        # save
        output = open(out_dir + '/' + dataset_name + '.pkl', 'wb')
        pickle.dump(clf, output)

        y_val = data['Y_val']
        X_val = ConvLSTMAE.embeddings(args, svm_data, data['X_val'], model)

        y_pred = clf.predict(X_val)
        accuracy = np.mean(y_pred == y_val)
        print 1 - accuracy <= Best_Results[dataset_name], 1 - accuracy <= Timenet_Results[
            dataset_name], dataset_name, 1 - accuracy, Timenet_Results[dataset_name], Best_Results[dataset_name]
