import ROOT
import tensorflow as tf



ROOT.gSystem.Load('/sps/nemo/scratch/amendl/AI/real_testing/TKEventEnvironment/TKEvent/lib/libTKEvent.so')


def generator(start,end,run_number):
    '''
    
    '''
    for i in range(start,end):
        yield tf.constant([i,run_number],dtype=tf.int64)


def py_load_event(o):
    pass
def py_load_event_height(o):
    pass
def py_load_event_front(o):
    pass
def py_load_result(o):
    pass

def load_event_helper(event_id):
    """
    
    """
    file = ROOT.TFile("/sps/nemo/scratch/amendl/AI/real_testing/runs/Run-%i.root" % (event_id[1]))
    tree = file.Get('Event')
    tree.GetEntry(event_id[0])
    
    # TODO

    event                                  = py_load_event(tree.Eventdata)
    event_height                           = py_load_event_height(tree.Eventdata)
    event_front                            = py_load_event_front(tree.Eventdata)


def load_event(event_id):
    """

    """
    [event_top,event_height,event_front,result,result_row,result_column]    =   tf.py_function(
                                                                                    func=load_event_helper,
                                                                                    inp=[event_id],
                                                                                    Tout=[
                                                                                        tf.TensorSpec(shape=(9,113),    dtype=tf.float64),
                                                                                        tf.TensorSpec(shape=(30,113),   dtype=tf.float64),
                                                                                        tf.TensorSpec(shape=(30,9),     dtype=tf.float64),
                                                                                        tf.TensorSpec(shape=(4),        dtype=tf.float64),
                                                                                        tf.TensorSpec(shape=(15),       dtype=tf.float64),
                                                                                        tf.TensorSpec(shape=(22),       dtype=tf.float64)
                                                                                    ]
                                                                                )
                 
    event_top           .set_shape((9,113))
    event_height        .set_shape((30,113))
    event_front         .set_shape((30,9))

    result              .set_shape((4))
    result_row          .set_shape((15))
    result_column       .set_shape((22))

    #return (event_top,event_height,event_front),(result,result_row,result_column)
    return event_front,result