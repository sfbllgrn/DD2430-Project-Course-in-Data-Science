# resnet model
from socketserver import ThreadingUDPServer
import keras
from keras import layers
import numpy as np
import time

import keras.backend as K

from InceptionTime.utils import calculate_metrics

# NILS NOTE: These functions are not used anymore since there were problems with saving/loading
# them! Instead, keras.metrics.function are used in the model.compile function!
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


class Classifier_TRANSFORMER:

    def __init__(self, checkpoints_path, input_shape, nb_classes, save_weights=True,verbose=False, batch_size=64,
                 head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25, lr=0.001, wd=None, early_stop=True):

        self.early_stop = early_stop
        self.lr = lr
        self.wd = wd
        self.checkpoints_path = checkpoints_path
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.mlp_dropout = mlp_dropout
        self.dropout = dropout
        self.callbacks = None
        self.batch_size = batch_size
        self.save_weights = save_weights
        self.verbose = verbose
        self.model = self.build_model(input_shape, nb_classes)


        if self.save_weights:
            self.model.save_weights(self.checkpoints_path)


    def _transformer_encoder(self, inputs):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(
            key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout
        )(x, x)
        x = layers.Dropout(self.dropout)(x)
        res = x + inputs
    
        # Feed Forward Part
        x = layers.LayerNormalization(epsilon=1e-6)(res)
        x = layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu")(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        return x + res

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.Input(shape=input_shape)
        x = input_layer
        for _ in range(self.num_transformer_blocks):
            x = self._transformer_encoder(x)
    
        x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        for dim in self.mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(self.mlp_dropout)(x)
        output_layer = layers.Dense(nb_classes, activation="softmax")(x)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                       metrics=['accuracy', keras.metrics.Recall(), keras.metrics.Precision(), keras.metrics.AUC(name='f1_score')]

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
