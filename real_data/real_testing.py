import ROOT
import tensorflow as tf
from tensorflow import keras
import numpy as np

import tkevent_dataset_pipeline as task







if __name__ == "__main__":

    number_of_classes   = 4

    run_number          = 974
    events_start        = 0
    events_end          = 100000
    file                = "model"

    model = keras.models.load_model(file)

    dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(events_start,events_end,run_number),
        output_signature=(tf.TensorSpec(shape=(2),dtype=tf.int64))
    ).map(task.load_event).batch(256).prefetch(1)

    matrix = np.zeros((number_of_classes+2,number_of_classes+2))
    

    for dataseto in dataset:
        matrix[np.argmax(model(dataseto[0])),dataseto[1]]+=1


    print(matrix)
