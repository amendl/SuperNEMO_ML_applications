# Tricks for TensorFlow
 * [`tf.keras.utils.disable_interactive_logging`](https://www.tensorflow.org/api_docs/python/tf/keras/utils/disable_interactive_logging)
 * [`tf.keras.callbacks.ModelCheckpoint`](https://saturncloud.io/blog/saving-your-best-model-in-keras-a-comprehensive-guide/)
 * [`map` `interleave` and `flat_map`](https://stackoverflow.com/questions/66778153/how-exactly-does-tf-data-dataset-interleave-differ-from-map-and-flat-map)
```
num_of_cores = multiprocessing.cpu_count() # num of available cpu cores
mapped_data = data.map(function, num_parallel_calls = num_of_cores)
```






# Documentation
### matteo_without_skip_connection
best found - loss: 0.0108 - accuracy: 0.9962
final - loss: 0.0306 - accuracy: 0.9879
### matteo_small_latent_space_Adam
best yet (2nd epoch) - loss: 0.0367 - accuracy: 0.9832
final -
### matteo_small_latent_space_SGD (learning rate 0.00001)
 * will not converge better than Adam 
best yet
final
### matteo_small_latent_spaceSGDs (learning rate 0.000001)
 * will not converge better than Adam