# resnet model
from socketserver import ThreadingUDPServer
import keras
import numpy as np
import time

import keras.backend as K

from InceptionTime.utils import calculate_metrics

def Recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall

def Precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision    

def F1_score(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


@keras.saving.register_keras_serializable()
class Classifier_INCEPTION:

    def __init__(self, checkpoints_path, input_shape, nb_classes, save_weights=True,verbose=False, batch_size=64,
                 nb_filters=32, use_residual=True, use_bottleneck=True, depth=6, kernel_size=41, lr=0.001, wd=None, early_stop=True):

        self.early_stop = early_stop
        self.lr = lr
        self.wd = wd
        self.checkpoints_path = checkpoints_path
        self.nb_filters = nb_filters
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        self.depth = depth
        self.kernel_size = kernel_size - 1
        self.callbacks = None
        self.batch_size = batch_size
        self.bottleneck_size = 32
        self.save_weights = save_weights
        self.verbose = verbose
        self.model = self.build_model(input_shape, nb_classes)

        if self.save_weights:
            self.model.save_weights(self.checkpoints_path)


    def _inception_module(self, input_tensor, stride=1, activation='linear'):
        if self.use_bottleneck and int(input_tensor.shape[-1]) > 1:
            input_inception = keras.layers.Conv1D(filters=self.bottleneck_size, kernel_size=1,
                                                  padding='same', activation=activation, use_bias=False)(input_tensor)
        else:
            input_inception = input_tensor
        # kernel_size_s = [3, 5, 8, 11, 17]
        kernel_size_s = [self.kernel_size // (2 ** i) for i in range(3)]
        conv_list = []
        for i in range(len(kernel_size_s)):
            conv_list.append(keras.layers.Conv1D(filters=self.nb_filters, kernel_size=kernel_size_s[i],
                                                 strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))

        max_pool_1 = keras.layers.MaxPool1D(pool_size=3, strides=stride, padding='same')(input_tensor)
        conv_6 = keras.layers.Conv1D(filters=self.nb_filters, kernel_size=1,
                                     padding='same', activation=activation, use_bias=False)(max_pool_1)
        conv_list.append(conv_6)
        x = keras.layers.Concatenate(axis=2)(conv_list)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation(activation='relu')(x)
        return x


    def _shortcut_layer(self, input_tensor, out_tensor):
        shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                         padding='same', use_bias=False)(input_tensor)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
        x = keras.layers.Add()([shortcut_y, out_tensor])
        x = keras.layers.Activation('relu')(x)
        return x


    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)
        x = input_layer
        input_res = input_layer
        for d in range(self.depth):
            x = self._inception_module(x)
            if self.use_residual and d % 3 == 2:
                x = self._shortcut_layer(input_res, x)
                input_res = x

        gap_layer = keras.layers.GlobalAveragePooling1D()(x)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)
        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=self.lr, epsilon=0.1, weight_decay=self.wd),
                      metrics=['accuracy', F1_score, Precision, Recall])
        
        # Callbacks
        self.callbacks = [keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=50, min_lr=1e-6)]
        if self.early_stop:
            early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-3, patience=50, restore_best_weights=True)
            self.callbacks.append(early_stopping)

        if self.save_weights:
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=self.checkpoints_path, monitor='val_loss', save_best_only=True)
            self.callbacks.append(model_checkpoint)

        return model


    def fit(self, x_train, y_train, x_val, y_val, nb_epochs, plot_test_acc=True):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        if self.batch_size is None:
            mini_batch_size = int(min(x_train.shape[0] / 10, 16))
        else:
            mini_batch_size = self.batch_size

        if plot_test_acc:
            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                                  verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        else:
            hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                                  verbose=self.verbose)
        return hist
   

    def predict(self, x_test, use_best_model=True):
        if use_best_model:
            model = keras.models.load_model(self.checkpoints_path) 
        else:
            model = self.model
        y_pred = model.predict(x_test, batch_size=self.batch_size)
        return y_pred
