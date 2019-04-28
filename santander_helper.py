import tensorflow as tf
from sklearn.metrics import roc_auc_score
import keras
import numpy as np
tf.logging.set_verbosity(tf.logging.ERROR)

def auc(y_true, y_pred):
    f1 = lambda: tf.constant(0.5, dtype=tf.float64)
    f2 = lambda: tf.py_function(roc_auc_score, (y_true, y_pred), tf.double)
    
    r = tf.case([(tf.equal(tf.reduce_sum(y_true), tf.constant(0.5, dtype=tf.float32)), f1),
                 (tf.equal(tf.reduce_sum(tf.subtract(tf.ones_like(y_true), y_true)), tf.constant(0.5, dtype=tf.float32)), f1)
                ], default=f2)
    return r

class DataGenerator(keras.utils.Sequence):
    @staticmethod
    def augment(x, y, class_1_aug=2, class_0_aug=1, common_rows = 2):
        # Only works for binary targets
        xs,xn = [],[]
        orig_shape = x.shape
        x_aug = x.copy()
        y_aug = y.copy()

        for i in range(class_1_aug):
            x1 = x_aug[y_aug==1].copy()
            ids = np.arange(x1.shape[0])
            for c in range(x1.shape[1]):
                np.random.shuffle(ids)
                x1[:,c*common_rows:c*common_rows+common_rows] = x1[:,c*common_rows:c*common_rows+common_rows][ids]
            xs.append(x1)

        for i in range(class_0_aug):
            x1 = x_aug[y_aug==0].copy()
            ids = np.arange(x1.shape[0])
            for c in range(x1.shape[1]):
                np.random.shuffle(ids)
                x1[:,c*common_rows:c*common_rows+common_rows] = x1[:,c*common_rows:c*common_rows+common_rows][ids]
            xn.append(x1)

        xs = np.vstack(xs)

        xn = np.vstack(xn)
        ys = np.ones(len(xs))
        yn = np.zeros(len(xn))

        return np.vstack([xs,xn]), np.concatenate([ys,yn])
    def __init__(self, X_train, y_train, batch_size=512, shuffle=True, class_1_aug=1, class_0_aug=1, common_rows=2, random_seed=42):
        np.random.seed(random_seed)
        self.batch_size = batch_size
        self.X_train = X_train
        self.y_train = y_train
        self.X_train_aug = None
        self.y_train_aug = None
        self.shuffle = shuffle
        self.class_1_aug = class_1_aug
        self.class_0_aug = class_0_aug
        self.common_rows = common_rows
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        return self.X_train_aug[indexes], self.y_train_aug[indexes]
        # return self.X_train_aug[indexes], self.y_train_aug[indexes].reshape(-1,1)*np.ones((len(self.y_train_aug[indexes]), 200))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.X_train_aug, self.y_train_aug = DataGenerator.augment(self.X_train, self.y_train, 
                                                     class_1_aug=self.class_1_aug, 
                                                     class_0_aug=self.class_0_aug,
                                                     common_rows = self.common_rows
                                                    )
        self.indexes = np.arange(len(self.X_train_aug))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)