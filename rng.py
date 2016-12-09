from numpy.random import RandomState
import tensorflow as tf

seed = 12
np_rng = RandomState(seed)
tf.set_random_seed(seed)
