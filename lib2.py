import tensorflow as tf
from tensorflow import keras
from inspect import currentframe



class RandomCell:
    '''
    
    '''
    def __init__(self,side,row,layer,rate,fire,distribution_):
        '''
            Generates noise for specific cell
        '''
        if (self.fire != True and self.fire != False) or (side != 0 and side != 1) or row <0 or row>9 or layer<0 or layer >113 :
            raise Exception(f"Custom exception in {__file__}:{currentframe().f_back.f_locals}: side = {side}, row = {row}, layer = {layer}, rate = {rate}, fire = {fire}")
        
        self.side   = side
        self.row    = row
        self.layer  = layer
        self.fire   = fire
        self.dist   = distribution_
        self.rate   = rate

    def __call__(self,top_projection,side_projection,front_projection,side):
        '''
        
        '''
        if side==2 and side==self.side:
            fill = 0. if self.fire == False else 1. 
            z = int((max(min(self.dist(),1490.),-1500.)+1500.)/100.)
            top_projection[self.layer,self.row]         = fill
            side_projection[z,self.row]                 = fill
            front_projection[z,self.layer]              = fill



class RandomFullDetector():
    '''
        Generates noise for all detector
    '''
    def __init__(self,rate,fire,distribution_):
        '''
        '''
        self.rate = rate
        self.fire = fire
        self.dist = distribution_

    def __call__(self,top_projection,side_projection,front_projection,side):
        '''
        
        '''
        if not tf.random.uniform([]).numpy() < self.rate:
            fill    = 0. if self.fire == False else 1. 
            layer   = int(tf.random.uniform([],minval=0,maxval=9,dtype=tf.dtypes.int32))
            row     = int(tf.random.uniform([],minval=0,maxval=113,dtype=tf.dtypes.int32))
            z = int((max(min(self.dist(),1490.),-1500.)+1500.)/100.)
            top_projection[layer,row]     = fill
            side_projection[z,row]             = fill
            front_projection[z,layer]          = fill
            self(top_projection,side_projection,front_projection,side)
            

    

            





