import math
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer, Softmax, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import Constant, glorot_normal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.losses import BinaryCrossentropy
from matplotlib import pyplot as plt


class ConcreteSelect(Layer):

    def __init__(self, output_dim, start_temp=10.0, min_temp=0.1, alpha=0.99999, thresh_alpha=0.999,
                 initial_weights=None, **kwargs):
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.min_temp = K.constant(min_temp)
        self.alpha = K.constant(alpha)
        self.thresh_alpha = K.constant(thresh_alpha)
        self.initial_weights = initial_weights
        super(ConcreteSelect, self).__init__(**kwargs)

    def build(self, input_shape):
        self.temp = self.add_weight(name='temp', shape=[], initializer=Constant(self.start_temp), trainable=False)
        self.thresh = self.add_weight(name='thresh', shape=[], initializer=Constant(3.0), trainable=False)
        if self.initial_weights is None:
            self.initial_weights = glorot_normal()
        else:
            self.initial_weights = Constant(np.asarray(self.initial_weights))
        self.logits = self.add_weight(name='logits', shape=[self.output_dim, input_shape[1]],
                                      initializer=self.initial_weights, trainable=True)
        super(ConcreteSelect, self).build(input_shape)

    def call(self, X, training=None):
        uniform = K.random_uniform(self.logits.shape, K.epsilon(), 1.0)
        gumbel = -K.log(-K.log(uniform))
        temp = K.update(self.temp, K.maximum(self.min_temp, self.temp * self.alpha))
        thresh = K.update(self.thresh, K.maximum(1.0, self.thresh * self.thresh_alpha))
        noisy_logits = (self.logits + gumbel) / temp
        samples = K.softmax(noisy_logits)

        discrete_logits = K.one_hot(K.argmax(self.logits), self.logits.shape[1])

        self.selections = K.in_train_phase(samples, discrete_logits, training)
        Y = K.dot(X, K.transpose(self.selections))

        return Y

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def regularization_loss(self):
        selections = K.softmax(self.logits)
        reg = 0.1 * K.sum(K.relu(K.sum(selections, axis=-1) - self.thresh))
        return reg


class StopperCallback(EarlyStopping):

    def __init__(self, mean_max_target=0.998):
        self.mean_max_target = mean_max_target
        super(StopperCallback, self).__init__(monitor='', patience=float('inf'), verbose=1, mode='max',
                                              baseline=self.mean_max_target)

    def on_epoch_begin(self, epoch, logs=None):
        print('mean max of probabilities:', self.get_monitor_value(logs), '- temperature',
              K.get_value(self.model.get_layer('concrete_select').temp))
        print('regularization loss:', K.get_value(self.model.get_layer('concrete_select').regularization_loss()))
        # print( K.get_value(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis = -1)))
        # print(K.get_value(K.max(self.model.get_layer('concrete_select').selections, axis = -1)))

    def get_monitor_value(self, logs):
        monitor_value = K.get_value(K.mean(K.max(K.softmax(self.model.get_layer('concrete_select').logits), axis=-1)))
        return monitor_value


class ConcreteAutoencoderFeatureSelector:

    def __init__(self, K, output_function, num_epochs=300, batch_size=None, learning_rate=0.001, start_temp=10.0,
                 min_temp=0.1, tryout_limit=5, class_weights=None, initial_weights=None):
        self.K = K
        self.output_function = output_function
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.tryout_limit = tryout_limit
        self.class_weights = class_weights
        self.initial_weights = initial_weights

    def fit(self, X, Y=None, val_X=None, val_Y=None):
        if Y is None:
            Y = X
        assert len(X) == len(Y)
        validation_data = None
        if val_X is not None and val_Y is not None:
            assert len(val_X) == len(val_Y)
            validation_data = (val_X, val_Y)

        if self.batch_size is None:
            self.batch_size = max(len(X) // 256, 16)

        num_epochs = self.num_epochs
        steps_per_epoch = (len(X) + self.batch_size - 1) // self.batch_size

        for i in range(self.tryout_limit):

            K.set_learning_phase(1)

            inputs = Input(shape=X.shape[1:])

            alpha = math.exp(math.log(self.min_temp / self.start_temp) / (num_epochs * steps_per_epoch))
            thresh_alpha = math.exp(math.log(1.0 / 3.0) / (num_epochs * steps_per_epoch))

            self.concrete_select = ConcreteSelect(self.K, self.start_temp, self.min_temp, alpha, thresh_alpha,
                                                  initial_weights=self.initial_weights,
                                                  name='concrete_select')

            selected_features = self.concrete_select(inputs)

            outputs = self.output_function(selected_features)

            if self.class_weights is not None:
                self.class_weights = dict(zip(np.arange(0, X.shape[1]), self.class_weights))
            else:
                self.class_weights = dict(zip(np.arange(0, X.shape[1]), np.ones(1, X.shape[1])))

            self.model = Model(inputs, outputs)

            self.model.compile(Adam(self.learning_rate),
                               loss=lambda y_true, y_pred: BinaryCrossentropy(from_logits=False)(y_true,
                                                                                                 y_pred) + self.concrete_select.regularization_loss(),
                               metrics=[BinaryAccuracy()])
            # self.model.compile(Adam(self.learning_rate), loss='kullback_leibler_divergence',
            # metrics=[BinaryAccuracy()])

            print(self.model.summary())

            stopper_callback = StopperCallback()

            hist = self.model.fit(X, Y, self.batch_size, num_epochs, verbose=1,
                                  callbacks=[stopper_callback], validation_data=validation_data,
                                  class_weight=self.class_weights)  # , validation_freq = 10)

            plt.figure()
            # summarize history for accuracy
            plt.plot(hist.history['binary_accuracy'])
            plt.plot(hist.history['val_binary_accuracy'])
            plt.title('model accuracy')
            plt.ylabel('binary accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(hist.history['loss'])
            plt.plot(hist.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()

            if K.get_value(
                    K.mean(K.max(K.softmax(self.concrete_select.logits, axis=-1)))) >= stopper_callback.mean_max_target:
                break

            num_epochs *= 2

        self.probabilities = K.get_value(K.softmax(self.model.get_layer('concrete_select').logits))
        self.indices = K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

        return self.model

    def get_indices(self):
        return K.get_value(K.argmax(self.model.get_layer('concrete_select').logits))

    def get_mask(self):
        return K.get_value(K.sum(K.one_hot(K.argmax(self.model.get_layer('concrete_select').logits),
                                           self.model.get_layer('concrete_select').logits.shape[1]), axis=0))

    def transform(self, X):
        return X[self.get_indices()]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices=False):
        return self.get_indices() if indices else self.get_mask()

    def get_params(self):
        return self.model
