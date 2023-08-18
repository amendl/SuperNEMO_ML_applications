#! /sps/nemo/scratch/amendl/AI/virtual_env_python391/bin/python

import tensorflow as tf
from tensorflow import keras
import numpy as np
import ROOT
import matplotlib.pyplot as plt
import sys

def import_arbitrary_module(module_name,path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(module_name,path)
    imported_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = imported_module
    spec.loader.exec_module(imported_module)

    return imported_module

task = import_arbitrary_module("task","/sps/nemo/scratch/amendl/AI/my_lib/latent_space_tricks/VAE/my_dataset_with_hint.py")

def tf_to_numpy(tensor,matteo_shape=True):
    def non_vectorized_f(x):
        if x>0.5:
            return 1.
        else:
            return 0.
    vectorized_f = np.vectorize(non_vectorized_f)

    dataset = tf.data.Dataset.from_tensor_slices(tensor)
    iterator = dataset.as_numpy_iterator()
    elements = [element for element in iterator]
    x = np.vstack(elements)
    if matteo_shape:
        x = np.transpose(x)

        x = np.delete(x, (0,1,11), axis=0)
        x = np.delete(x,(0,1), axis=1)
    # x = vectorized_f(x)
    return x


def my_print(tensor):
    # print(tensor.shape)
    for i in range(116):
        print(chr(9608),end="")
    print()
    for i in range(tensor.shape[0]):
        print(chr(9608),end="")
        for j in range(tensor.shape[1]):
            if tensor[i,j]>0.545:
                print(chr(0x2299),end="")
            else:
                print(" ",end="")
        print(chr(9608))
    for i in range(116):
        print(chr(9608),end="")
    

class ThresholdFinder:
    '''
    
    '''
    def __init__(self):
        self.histo = np.zeros((100))
    def fill(self, original,truth,model_output):
        for i in range(100):
            threshold = 0.005 + float(i)*0.01
            for j in range(model_output.shape[0]):
                for k in range(model_output.shape[1]):
                    if original[j,k]>0.5:
                        if (model_output[j,k] > threshold) == (truth[j,k]>0.5):
                            self.histo[i]+=1
                        else:
                            self.histo[i]-=1

class SegmentationMetrics:
    '''
    
    '''
    def __init__(self,histo_sizes,threshold):
        self.threshold = threshold
        self.histo_sizes = histo_sizes
        self.less_than_expected = []
        self.more_than_expected = []
        self.randomly_added     = []
    def fill(self, original,truth,model_output):
        n_of_preserved_truth  = 0
        n_of_removed_truth    = 0
        less_than_expected    = 0
        more_than_expected    = 0
    
        for j in range(model_output.shape[0]):
            for k in range(model_output.shape[1]):
                if original[j,k] > 0.5:
                    # truth
                    if truth[j,k] > 0.5:
                        n_of_preserved_truth+=1
                    if truth[j,k] < 0.5:
                        n_of_removed_truth+=1
                    # model
                    if model_output[j,k] < self.threshold and truth[j,k] > 0.5:
                        less_than_expected+=1
                    if model_output[j,k] > self.threshold and truth[j,k] < 0.5:
                        more_than_expected+=1
        # fill
        self.less_than_expected.append(float(less_than_expected)/float(n_of_preserved_truth))
        self.more_than_expected.append(float(more_than_expected)/float(n_of_removed_truth))

    def plot_less_than_expected(self,params):
        plt.clf()
        plt.hist(self.less_than_expected,bins=self.histo_sizes)
        plt.title("Less than expected")
        plt.savefig(params)

    def plot_more_than_ecpected(self,params):
        plt.clf()
        plt.hist(self.more_than_expected,bins=self.histo_sizes)
        plt.title("More than expected")
        plt.savefig(params)

    def plot_randomly_added(self,params):
        raise NotImplementedError

def analyse_event(ID,model,fillers):
    original,truth = task.load_event_helper(ID)
    model_output = model(tf.reshape(original,[1,9,113]))
    # for f in fillers:
    #     f.fill(tf_to_numpy(original),tf_to_numpy(truth),tf_to_numpy(model_output))
    my_print(tf_to_numpy(tf.reshape(original,[9,113]),False))
    print()
    my_print(tf_to_numpy(tf.reshape(model_output,[116,12])))
    print()





if __name__ == "__main__":
    model = keras.models.load_model("../VAE_test2/model")
    # model = keras.models.load_model("../matteo_without_skip/model")

    finder  = ThresholdFinder()
    metrics = SegmentationMetrics(10,0.545)
    print(model)
    for i in range(50): 
        print(f'2,2,{i}')
        analyse_event(tf.constant([2,2,i]),model,[metrics])

    #     if i % 10 == 0:
    #         print(i,flush=True)
    #         sys.stdout.flush()
    # metrics.plot_less_than_expected("less_than_expected.pdf")
    # metrics.plot_more_than_ecpected("more_than_expected.pdf")


    #     if i % 10 == 0:
    #         print(i,flush=True)

    # print(0.005+0.01*np.argmax(finder.histo))
    # plt.plot(np.linspace(0.005,1.-0.005,num=100),finder.histo)
    # plt.axvline(0.005+0.01*float(np.argmax(finder.histo)),color="red",linestyle="dashed")
    # plt.xlabel("threshold")
    # plt.ylabel("score (the bigger the better)")
    # plt.savefig("find.pdf")    
