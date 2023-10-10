from pyts.datasets import ucr_dataset_list

DATASET_NAMES = ucr_dataset_list()[7:9]

from utils import read_all_datasets
from utils import transform_labels
from utils import create_directory
from utils import generate_results_csv

import utils
import numpy as np
import sys
import sklearn



def prepare_data(dataset_obj):
    x_train = dataset_obj['data_train']
    y_train = dataset_obj['target_train']
    x_test = dataset_obj['data_test']
    y_test = dataset_obj['target_test']

    # x_train = datasets_dict[dataset_name]['data_train']
    # y_train = datasets_dict[dataset_name]['target_train']
    
    # x_test = datasets_dict[dataset_name]['data_test']
    # y_test = datasets_dict[dataset_name]['target_test']

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # make the min to zero of labels
    y_train, y_test = transform_labels(y_train, y_test)

    # save orignal y because later we will use binary
    y_true = y_test.astype(np.int64)
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

    return x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc


def fit_classifier():
    input_shape = x_train.shape[1:]

    classifier = create_classifier(classifier_name, input_shape, nb_classes,
                                   output_directory, verbose=True)

    classifier.fit(x_train, y_train, x_val, y_val, y_true, plot_test_acc=True)


def create_classifier(classifier_name, input_shape, nb_classes, output_directory,
                      verbose=False, build=True):

    N_EPOCHS = 100
    if classifier_name == 'nne':
        import nne
        return nne.Classifier_NNE(output_directory, input_shape,
                                  nb_classes, verbose)
    if classifier_name == 'inception':
        import InceptionModule_keras
        return InceptionModule_keras.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose,
                                              build=build, nb_epochs=N_EPOCHS)


def get_xp_val(xp):
    if xp == 'batch_size':
        xp_arr = [16, 32, 128]
    elif xp == 'use_bottleneck':
        xp_arr = [False]
    elif xp == 'use_residual':
        xp_arr = [False]
    elif xp == 'nb_filters':
        xp_arr = [16, 64]
    elif xp == 'depth':
        xp_arr = [3, 9]
    elif xp == 'kernel_size':
        xp_arr = [8, 64]
    else:
        raise Exception('wrong argument')
    return xp_arr


############################################### main
import os
root_dir = os.path.dirname(os.path.dirname(os.getcwd()))

xps = ['use_bottleneck', 'use_residual', 'nb_filters', 'depth',
       'kernel_size', 'batch_size']

if sys.argv[1] == 'InceptionTime':
    # run nb_iter_ iterations of Inception on the whole TSC archive
    classifier_name = 'inception'
    nb_iter_ = 5

  
    #datasets_dict = read_all_datasets(root_dir)

    import load_data
    datasets_dict = {}
    CACHED_DATA_FOLDER = os.path.dirname(os.path.dirname(os.getcwd())) + "/Data"
    for dataset_name in DATASET_NAMES:
        cache_path = os.path.join(CACHED_DATA_FOLDER, dataset_name)
        dataset_obj = load_data.fetch_ucr_dataset(dataset=dataset_name, use_cache=True, data_home=cache_path)
        datasets_dict[dataset_name] = dataset_obj


    for iter in range(nb_iter_):
        print('\t\titer', iter)

        trr = ''
        if iter != 0:
            trr = '_itr_' + str(iter)

        tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + trr + '/'

        for dataset_name in DATASET_NAMES:
            print('\t\t\tdataset_name: ', dataset_name)

            dataset_obj = datasets_dict[dataset_name]

            x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data(dataset_obj)
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

    datasets_dict = read_all_datasets(root_dir)

    tmp_output_directory = root_dir + '/results/' + classifier_name  + '/'

    for dataset_name in DATASET_NAMES:
        print('\t\t\tdataset_name: ', dataset_name)

        x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

        output_directory = tmp_output_directory + dataset_name + '/'

        fit_classifier()

        print('\t\t\t\tDONE')


elif sys.argv[1] == 'InceptionTime_xp':
    # this part is for running inception with the different hyperparameters
    # listed in the paper, on the whole TSC archive
    classifier_name = 'inception'
    max_iterations = 5

    datasets_dict = read_all_datasets(root_dir)

    for xp in xps:

        xp_arr = get_xp_val(xp)

        print('xp', xp)

        for xp_val in xp_arr:
            print('\txp_val', xp_val)

            kwargs = {xp: xp_val}

            for iter in range(max_iterations):

                trr = ''
                if iter != 0:
                    trr = '_itr_' + str(iter)
                print('\t\titer', iter)

                for dataset_name in DATASET_NAMES:

                    output_directory = root_dir + '/results/' + classifier_name + '/' + '/' + xp + '/' + '/' + str(
                        xp_val) + '/' + trr + '/' + dataset_name + '/'

                    print('\t\t\tdataset_name', dataset_name)
                    x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

                    # check if data is too big for this gpu
                    size_data = x_train.shape[0] * x_train.shape[1]

                    temp_output_directory = create_directory(output_directory)

                    if temp_output_directory is None:
                        print('\t\t\t\t', 'Already_done')
                        continue

                    input_shape = x_train.shape[1:]

                    from classifiers import inception

                    classifier = inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes,
                                                                verbose=False, build=True, **kwargs)

                    classifier.fit(x_train, y_train, x_test, y_test, y_true)

                    # the creation of this directory means
                    create_directory(output_directory + '/DONE')

                    print('\t\t\t\t', 'DONE')

    # we now need to ensemble each iteration of inception (aka InceptionTime)

    classifier_name = 'nne'

    datasets_dict = read_all_datasets(root_dir)

    tmp_output_directory = root_dir + '/results/' + classifier_name + '/'

    for xp in xps:
        xp_arr = get_xp_val(xp)
        for xp_val in xp_arr:

            clf_name = 'inception/' + xp + '/' + str(xp_val)

            for dataset_name in DATASET_NAMES:
                x_train, y_train, x_test, y_test, y_true, nb_classes, y_true_train, enc = prepare_data()

                output_directory = tmp_output_directory + dataset_name + '/'

                from classifiers import nne

                classifier = nne.Classifier_NNE(output_directory, x_train.shape[1:],
                                                nb_classes, clf_name=clf_name)

                classifier.fit(x_train, y_train, x_test, y_test, y_true)


elif sys.argv[1] == 'generate_results_csv':
    clfs = []
    itr = '-0-1-2-3-4-'
    inceptionTime = 'nne/inception'
    # add InceptionTime: an ensemble of 5 Inception networks
    clfs.append(inceptionTime + itr)
    # add InceptionTime for each hyperparameter study
    for xp in xps:
        xp_arr = get_xp_val(xp)
        for xp_val in xp_arr:
            clfs.append(inceptionTime + '/' + xp + '/' + str(xp_val) + itr)
    df = generate_results_csv('results.csv', root_dir, clfs)
    print(df)
