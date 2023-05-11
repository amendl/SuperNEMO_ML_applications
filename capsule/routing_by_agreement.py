'''
Dynamic loop for routing by agreement https://arxiv.org/abs/1710.09829

'''

import tensorflow as tf

class RoutingByAgreement():
    '''
        Routing by agreement
    '''
    __slots__ = 'iterations'
    def __init__(self,iterations):
        self.iterations=iterations
    def stop_condition(self,input,counter):
        return tf.less(counter,self.iterations)
    def loop_body(self,input,counter):
        return input # TODO
    def __call__(self,input):
        with tf.name_scope("routing_by_agreement_operation"):
            counter = 0
            result = tf.while_loop(self.stop_condition,self.loop_body,[input,counter])
            return result
