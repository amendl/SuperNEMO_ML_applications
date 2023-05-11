'''
    Implementation of fundamental parts of CapsNET https://arxiv.org/abs/1710.09829
'''

import tensorflow as tf
from tensorflow import keras


from routing_by_agreement import RoutingByAgreement;



# class CapsuleNetwork(keras.Model):
#     def __init__(
#             self,
#             input_size,
#             no_of_conv_kernels,
#             conv_strides,


            
            
#             )

#         self.epsilon=1e-7
#         self.input_size=input_size
#         self.number_of_conv_kernels



@tf.function
def squash(s, axis=-1,epsilon=1e-7):
    '''
        Activation function for capsule. 
        For more information, see for example:
         - https://arxiv.org/abs/1710.09829
         - https://pechyonkin.me/capsules-2/
         - https://www.kaggle.com/code/giovanimachado/capsnet-tensorflow-implementation
    '''

    squared_norm = tf.reduce_sum(tf.square(s), axis=axis,
                                    keepdims=True)
    safe_norm = tf.sqrt(squared_norm + epsilon)
    squash_factor = squared_norm / (1. + squared_norm)
    unit_vector = s / safe_norm
    return squash_factor * unit_vector

class SecondaryCapsule(keras.layers.Layer):
    # 'capsules','affine_transform_matrix','routing_weights','capsules_input','features', 'input_spec'
    def __init__(
            self,
            capsules,

            transformation_initializer="glorot_uniform",
            routing_initializer="zeros",
            transformation_regularizer=None,
            routing_regularizer=None,
            activity_regularizer=None,
            transformation_constraint=None,
            routing_constraint=None,
            **kwargs
        ):
        super().__init__(activity_regularizer=activity_regularizer,**kwargs)

        self.capsules                           = capsules

        self.transformation_initializer         = transformation_initializer
        self.routing_initializer                = routing_initializer
        self.transformation_regularizer         = transformation_regularizer
        self.routing_regularizer                = routing_regularizer
        self.activity_regularizer               = activity_regularizer
        self.transformation_constraint          = transformation_constraint
        self.routing_constraint                 = routing_constraint
    
    def build(self,input_shape):
        if(len(input_shape)!=4):
            raise Exception("Invalid dimension of input shape")
        
        self.capsules_input                     = input_shape[2] 
        self.features                           = input_shape[3]

        print("Running with batch_size= ",input_shape[0])

        self.input_spec                         = keras.layers.InputSpec(shape=(None,1,self.capsules_input,self.features))

        self.affine_transform_matrix            = self.add_weight(
            "affine_transform_matrix",
            shape           = [], # TODO
            initializer     = self.transformation_initializer,
            regularizer     = self.transformation_regularizer,
            constraint      = self.transformation_constraint,
            trainable       = True
        )

        self.routing_weights                    = self.add_weight(
            "routing_weights",
            shape           = [self.capsules,self.capsules_input],
            initializer     = self.transformation_initializer,
            regularizer     = self.transformation_regularizer,
            constraint      = self.transformation_constraint,
            trainable       = True
        )

        


    def call(self,inputs):
        pass # TODO


    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "capsules"      : self.capsules,
                "initializer"   : tf.initializers.serialize   (self.transformation_initializer),
                "regularizer"   : tf.regularizers.serialize   (self.transformation_regularizer),
                "constraint"    : tf.constraints.serialize    (self.transformation_constraint)
            }
        )        