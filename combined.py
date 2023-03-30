import datasets.number_of_tracks as task

from sklearn.metrics import confusion_matrix

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary. 
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''
    
    config = model.get_config()
    old_to_new = {}
    new_to_old = {}
    
    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]
    
    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
    
    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
    
    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)
    
    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
    
    return new_model


def plot_train_val_accuracy(history,name = "training_accuracy.pdf"):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(
        fname = name,
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

def load_trained_model(file, index,prefix):
    original = keras.models.load_model(file)
    A = add_prefix(original,prefix)
    for layer in A.layers:
        layer.trainable=False
    print(A.layers[index])
    return (A,A.layers[index])

if __name__=='__main__':
    print(tf.config.list_physical_devices())

    tracks = 4
    files=10
    events = 5000

    A, A_model_output = load_trained_model("../big_model/model",-5,"top_")
    B, B_model_output = load_trained_model("../height_model2/model",-5,"side_")
    C, C_model_output = load_trained_model("../front_model2/model",-5,"front_")
    
    x = layers.Concatenate()([A_model_output.output,B_model_output.output,C_model_output.output])
    x = layers.Dense(384,activation='tanh',use_bias=False)(x)
    x = layers.Dense(256,activation='tanh',use_bias=False)(x)
    x = layers.Dense(128,activation='tanh',use_bias=False)(x)
    x = layers.Dense(64,activation='tanh',use_bias=False)(x)
    x = layers.Dense(4)(x)
    x = layers.Softmax()(x)
    model = keras.Model(inputs=[A.input, B.input,C.input], outputs=x)

    dataset_size = tracks*files*events
    dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(tracks,[0,1,2,3,4,5,6,7],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    dataset = dataset.shuffle(dataset_size,reshuffle_each_iteration = True)
    dataset = dataset.map(task.load_event)
    print(dataset)

    val_dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(tracks,[8],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    val_dataset = val_dataset.map(task.load_event)
    print(val_dataset)


    test_dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(tracks,[9],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    test_dataset = test_dataset.map(task.load_event)


    # compile and train model with freezed convolutional section
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics = ['accuracy']
    )
    history = model.fit(
        x = dataset.batch(256),
        epochs = 15,
        validation_data = val_dataset.batch(256),
        callbacks=[EarlyStopping(monitor='val_accuracy',mode='max',baseline=0.9,start_from_epoch=5,min_delta=0.01)]
    )
    plot_train_val_accuracy(history,"freezed_convolution.pdf")
    confusion(model,test_dataset)

    # unfreeze convolutional section and train model
    for layer in model.layers:
        layer.trainable=True
    model.compile(
        optimizer = 'adam',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics = ['accuracy']
    )
    history = model.fit(
        x = dataset.batch(256),
        epochs = 15,
        validation_data = val_dataset.batch(256),
        callbacks=[EarlyStopping(monitor='val_accuracy',mode='max',baseline=0.9,start_from_epoch=5,min_delta=0.01)]
    )
    plot_train_val_accuracy(history,"unfreezed_convolution.pdf")
    confusion(model,test_dataset)

    model.save("model")
