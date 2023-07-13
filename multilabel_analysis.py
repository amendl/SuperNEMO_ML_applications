import tensorflow as tf
from tensorflow import keras
import numpy as np

import datasets.number_of_tracks_my_generator as task
import ROOT


if __name__=="__main__": 
    binary_crossentropy = True
    number_of_labels = 22
    tracks = [1,2,3,4]
    events = 5000

    binary_crossentropy_models_folders          = ["top_"+str(i)+"_labels" for i in tracks]
    cathegorical_crossentropy_models_folders    = ["top_1_labels","top_2_labels","top_3_labels","top_4_labels"] # TODO

    model_folders = binary_crossentropy_models_folders if binary_crossentropy == True else cathegorical_crossentropy_models_folders
    models = [keras.models.load_model("../"+folder+"/model") for folder in model_folders]

    datasets = [tf.data.Dataset.from_generator(
                    generator = lambda: task.generator(i,[9],events),
                    output_signature=(tf.TensorSpec(shape=(3),dtype=tf.int64))
                ) for i in tracks]
    predictions = [models[i](datasets[i].batch(512)) for i in tracks]

    correct_array = [np.zeros((number_of_labels)) for _ in tracks]
    uncorrect_correlations = np.zeros((number_of_labels,number_of_labels))

    for i in tracks:
        for event in range(events) :
            predicted_labels = np.argpartition(predictions[i][event],-i)[-i:]
            correct_labels = np.argpartition(TODO,-i)[-i:] # TODO
            same = 0
            for index1,index2 in range(i),range(i):
                if(predicted_labels[index1]==correct_labels[index2]):
                    same+=1
            correct_array[i][i-same]+=1
            if i-same == 1:
                # handle correlations
                predictedNotTrue = -1
                trueNotPredicted = -1
                for index1 in range(i):
                    is_present = False
                    for index2 in range(i):
                        if predicted_labels[index1] == correct_labels[index2]:
                            is_present = True
                    if is_present == False:
                        predictedNotTrue = index1
                        break;
                for index1 in range(i):
                    is_present = False
                    for index2 in range(i):
                        if predicted_labels[index2] == correct_labels[index1]:
                            is_present = True
                    if is_present == False:
                        truedNotPredicted = index1
                        break

    np.set_printoptions(threshold=np.inf)
    for i in correct_array:
        print(i)
    for i in uncorrect_correlations:
        print(i)
    
    output_file=ROOT.TFile("output_%i.root" % binary_crossentropy,"RECREATE")

    uncorrect_correlations_histogram = ROOT.TH2D("","",number_of_labels,-0.5,float(number_of_labels)-0.5,number_of_labels.-0.5,float(number_of_labels)-0.5)
    for index1,index2 in range(number_of_labels),range(number_of_labels):
        uncorrect_correlations_histogram->SetBinContent(index1+1,index2+1,uncorrect_correlations[index1][index2])
    uncorrect_correlations_histogram.Write()
    output_file.Close()