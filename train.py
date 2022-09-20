import keras
import tensorflow as tf
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from    keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
epochs =10

def get_model():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(512, 512, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (5, 5), activation='relu'), )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (5, 5), activation='relu'), )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (5, 5), activation='relu'), )
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    '''model.add(Dense(1024, activation='relu', name='fc1'))
    model.add(Dropout(0.2))'''
    model.add(Dense(120, activation='relu',name='fc1'))
    model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu',name='fc2'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2, beta_1=0.9, beta_2=0.999),

                  metrics=['accuracy'])
    return model

def load_data():
    file = open('pix.data', 'rb')

    # dump information to that file
    (pixels, labels) = pickle.load(file)

    # close the file
    file.close()

    print(pixels.shape)
    print(labels.shape)

    return pixels, labels

X, y = load_data()

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape)
print(Y_train.shape)

model = get_model()
filepath="weights_{epoch:02d}_{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]

aug = ImageDataGenerator()
model_hist = model.fit_generator(aug.flow(X_train, Y_train, batch_size=32),
                               epochs=epochs,# steps_per_epoch=len(X_train)//64,
                               validation_data=aug.flow(X_test,Y_test,
                               batch_size=32),
                               callbacks=callbacks_list)


model.save("modelsaved_5_07.h5")


