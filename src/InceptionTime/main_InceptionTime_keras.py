from pyts.datasets import ucr_dataset_list

DATASET_NAMES = ucr_dataset_list()[7:9]

from utils import transform_labels
from utils import create_directory

import utils
import numpy as np
import sys
import sklearn



def prepare_data(dataset_obj):
    x_train = dataset_obj['data_train']
    y_train = dataset_obj['target_train']
    x_test = dataset_obj['data_test']
    y_test = dataset_obj['target_test']

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true_test = y_test.astype(np.int64)
    y_true_train = y_train.astype(np.int64)
    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()


    if len(x_train.shape) == 2:  # if univariate
        # add a dimension to make it multivariate with one dimension
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    return x_train, y_train, x_test, y_test, y_true_test, nb_classes, y_true_train, enc


def fit_classifier():
    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes, verbose=True)

    classifier.fit(x_train, y_train, x_val, y_val, y_true_test, plot_test_acc=True)


def create_classifier(classifier_name, input_shape, nb_classes, verbose=False, build=True):

    N_EPOCHS = 100
    if classifier_name == 'nne':
        import nne
        return nne.Classifier_NNE(input_shape,
                                  nb_classes, verbose)
    
    if classifier_name == 'inception':
        import InceptionModule_keras
        return InceptionModule_keras.Classifier_INCEPTION(input_shape, nb_classes, verbose,
                                              build=build, nb_epochs=N_EPOCHS)



############################################### main
import os
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))

xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth',
       'kernel_size', 'batch_size']

if sys.argv[1] == 'InceptionTime':
    # run nb_iter_ iterations of Inception on the whole TSC archive
    classifier_name = 'inception'
    nb_iter_ = 5


    import load_data
    datasets_dict = {}
    CACHED_DATA_FOLDER = os.path.dirname(os.path.dirname(os.getcwd())) + "/Data"
    for dataset_name in DATASET_NAMES:
        cache_path = os.path.join(CACHED_DATA_FOLDER, dataset_name)
        dataset_obj = load_data.fetch_ucr_dataset(dataset=dataset_name, use_cache=True, data_home=cache_path)
        datasets_dict[dataset_name] = dataset_obj

    # Train the InceptionTime ensemble members 
    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)

        tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + trr + '/'

        for dataset_name in DATASET_NAMES:
            print('\t\t\tdataset_name: ', dataset_name)

            dataset_obj = datasets_dict[dataset_name]

            x_train, y_train, x_test, y_test, y_true_test, nb_classes, y_true_train, enc = prepare_data(dataset_obj)
            x_val = x_test
            y_val = y_test

            output_directory = tmp_output_directory + dataset_name + '/'

            temp_output_directory = create_directory(output_directory)

            if temp_output_directory is None:
                print('Already_done', tmp_output_directory, dataset_name)
                continue

            fit_classifier()

            print('\t\t\t\tDONE')

            # the creation of this directory means
            create_directory(output_directory + '/DONE')


    # run the ensembling of these iterations of Inception
    classifier_name = 'nne'

    tmp_output_directory = root_dir + '/results/' + classifier_name  + '/'

    for dataset_name in DATASET_NAMES:
        print('\t\t\tdataset_name: ', dataset_name)

        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

        output_directory = tmp_output_directory + dataset_name + '/'

        fit_classifier()

        print('\t\t\t\tDONE')

