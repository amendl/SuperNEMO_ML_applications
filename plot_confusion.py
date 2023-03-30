import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import datasets.number_of_tracks as task

def confusion(model, test_dataset,folder):
    '''
    
    '''
    index = 0
    event_index=0
    for input1,input2,input3,label in test_dataset:
        print(index)
        output = model(tf.reshape(input1,(1,9,113)),tf.reshape(input2,(1,30,113)),tf.reshape(input3,(1,30,9)))
        event_index=event_index+1
        if tf.argmax(label)!=np.argmax(output):
            plt.close()

            figure, axis = plt.subplots(3, 1)
            figure.suptitle(f"#{event_index} ... True: {label};\nPredicted: {output} ")
            axis[0].imshow(input1.numpy(),cmap='gray')
            axis[1].imshow(input2.numpy(),cmap='gray')
            axis[2].imshow(input3.numpy(),cmap='gray')
            figure.savefig(f"{folder}/{index}.pdf")
            index=index + 1

if __name__=='__main__':
    model = tf.keras.models.load_model("/sps/nemo/scratch/amendl/AI/my_lib/big_model/model")
    tracks=4
    events=1000
    test_dataset = tf.data.Dataset.from_generator(
        generator = lambda: task.generator(tracks,[8],events),
        output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
    )
    test_dataset = test_dataset.map(task.load_event)

    confusion(model,test_dataset,"/sps/nemo/scratch/amendl/AI/my_lib/big_model_testing")