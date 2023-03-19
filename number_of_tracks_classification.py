import datasets.number_of_tracks as task
from pp_cm import pp_matrix

from sklearn.metrics import confusion_matrix

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers # Dense, Droupout, Softmax, BatchNormalization, Conv2D
from keras.callbacks import EarlyStopping

import seaborn as sn
import matplotlib.pyplot as plt

def get_dataset_partitions(ds, ds_size, train_split=0.8, val_split=0.1):
    print(ds_size)
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

def architecture():
    '''

    '''
    i = keras.Input(shape=(9,113))

    x = layers.Reshape((9,113,1))(i)

    x = layers.Conv2D(4,3,activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (2,2))(x)

    x = layers.Conv2D(16,3,activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size=(2,2))(x)

    x = layers.Conv2D(64,3,activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(64,activation='relu',use_bias=True)(x)
    x = layers.Dense(4)(x)
    x = layers.Softmax()(x)

    model = keras.Model(inputs = i, outputs = x)

    return model

def architecture1():
    '''
    
    '''
    i = keras.Input(shape=(9,113))

    x = layers.Reshape((9,113,1))(i)

    x = layers.Conv2D(4,(3,10),activation = 'relu',padding="same")(x) # 4 
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)

    x = layers.Conv2D(16,(3,10),activation = 'relu',padding="same")(x) # 16
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)

    x = layers.Conv2D(64,(3,3),activation = 'relu',padding="same")(x) # 64
    x = layers.MaxPooling2D(pool_size = (2,2))(x)

    x = layers.Conv2D(128,(3,3),activation = 'relu',padding="same")(x) # 128
    x = layers.MaxPooling2D(pool_size = (2,2))(x)

    x = layers.Dropout(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128,activation='relu',use_bias=True)(x)
    x = layers.Dense(4)(x)
    x = layers.Softmax()(x)

    model = keras.Model(inputs = i, outputs = x)

    return model

def architecture2():
    '''
    
    '''
    i = keras.Input(shape=(9,113))
    img = layers.Reshape((9,113,1))(i)
    x = layers.Conv2D(16,(3,15),activation = 'relu',padding="same")(img)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)
    x = layers.Conv2D(128,(3,7),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) 
    x = layers.MaxPooling2D(pool_size = (2,2))(x) 
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation='relu',use_bias=True)(x)
    x = layers.Dense(64,activation='sigmoid',use_bias=True)(x)
    x = layers.Dense(4)(x)
    x = layers.Softmax()(x)

    model = keras.Model(inputs = i, outputs = x)

    return model

def architectureHeight():
    '''
    
    '''
    i = keras.Input(shape=(30,113))
    img = layers.Reshape((30,113,1))(i)
    x = layers.Conv2D(16,(3,7),activation = 'relu',padding="same")(img) # 16
    x = layers.MaxPooling2D(pool_size = (1,2),strides=(1,2))(x)
    x = layers.Conv2D(64,(3,3),activation = 'relu',padding="same")(x) # 64
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) # 128
    x = layers.MaxPooling2D(pool_size = (2,2))(x)
    x = layers.Conv2D(256,(3,3),activation = 'relu',padding="same")(x) # delete
    x = layers.MaxPooling2D(pool_size = (2,2))(x) # delete
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256,activation='relu',use_bias=True)(x)
    x = layers.Dense(64,activation='sigmoid',use_bias=True)(x)
    x = layers.Dense(4)(x)
    x = layers.Softmax()(x)

    model = keras.Model(inputs = i, outputs = x)

    return model

def architectureFront():
    '''
    
    '''
    i = keras.Input(shape=(30,9))
    img = layers.Reshape((30,9,1))(i)
    x = layers.Conv2D(16,(6,2),activation = 'relu',padding="same")(img)
    x = layers.Conv2D(64,(6,2),activation = 'relu',padding="same")(x)
    x = layers.MaxPooling2D(pool_size = (3,1),strides=(6,2))(x)
    x = layers.Conv2D(128,(2,2),activation = 'relu',padding="same")(x) # 64
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256,activation='relu',use_bias=True)(x)
    x = layers.Dense(64,activation='sigmoid',use_bias=True)(x)
    x = layers.Dense(4)(x)
    x = layers.Softmax()(x)
    
    model = keras.Model(inputs = i, outputs = x)

    return model




def plot_train_val_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(
        fname = 'training_accuracy.pdf',
        format = 'pdf'
    )

def confusion(model, test_dataset):
    y_true = []

    for _,label in test_dataset:
        y_true.append(tf.argmax(label))

    print("Creating confusion matrix")
    prediction=model.predict(test_dataset.batch(256))
    prediction = np.argmax(prediction, axis=1)
    cm = confusion_matrix(prediction, y_true)
    tf.print(cm,summarize=-1)
    # sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    # plt.savefig("confusion_matrix.pdf")


if __name__=="__main__":
    '''
    
    '''
    tracks = 4
    files=10
    events = 5000

    model = architectureFront()

    keras.utils.plot_model(model = model,to_file = "model.png",show_shapes= True,show_dtype = True)
    # visualizer(
    #     model = model,
    #     filename="graph.pdf",
    #     format='pdf',
    #     view = False
    # )
    
    dataset_size = tracks*files*events
    dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(tracks,[0,1,2,3,4,5,6,7],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    dataset = dataset.shuffle(dataset_size,reshuffle_each_iteration = True)
    dataset = dataset.map(task.load_event)
    print(dataset)

    # for i in dataset:
    #     tf.print(i,summarize=-1)

    val_dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(tracks,[8],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    val_dataset = val_dataset.map(task.load_event)

    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
    )
    history = model.fit(
        x = dataset.batch(256),
        epochs = 15,
        validation_data = val_dataset.batch(256),
        callbacks=[EarlyStopping(monitor='val_accuracy',mode='max',baseline=0.9,start_from_epoch=6,min_delta=0.01)]
    )
    plot_train_val_accuracy(history)

    test_dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(tracks,[9],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    test_dataset = test_dataset.map(task.load_event)


    confusion(model,test_dataset)

    model.save("model")
