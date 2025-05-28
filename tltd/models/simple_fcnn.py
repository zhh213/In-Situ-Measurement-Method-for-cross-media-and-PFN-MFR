from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout

from keras.regularizers import l2


def fully_fcnn(hidden_layers=[254, 128, 64], dropout_rate=0.1,
               l2_penalty=0, optimizer='adam',
               n_class=2):
    """
    Simple keres model for hyper-parameter tuning
    """
    loss = 'categorical_crossentropy'

    model = Sequential()
    for layers in hidden_layers:
        model.add(Dense(layers, activation='relu', kernel_regularizer=l2(l2_penalty)))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())

    model.add(Dense(n_class, activation='softmax'))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

    return model
