import tensorflow as tf

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

class NeuralNet:
    def __init__(self, layers, sizes):
        sess = tf.InteractiveSession()

        self.x = tf.placeholder(tf.float32, shape=[None, sizes[0]])
        self.y_ = tf.placeholder(tf.float32, shape=[None, sizes[-1]])

        self.weight_list = []
        self.bias_list = []
        self.fc = []

        for i in range(1, layers):
            self.weight_list += weight_variable([sizes[i], sizes[i+1]])
            self.bias_list += bias_variable([sizes[i+1]])

            prev_vals = self.x if i == 0 else self.fc[i-1]
            if i != layers - 1:
                self.fc += tf.nn.sigmoid(tf.matmul(prev_vals, self.weight_list[-1]) + self.bias_list[-1])
            else:
                self.fc += tf.matmul(prev_vals, self.weight_list[-1]) + self.bias_list[-1]

        sess.run(tf.global_variables_initializer())

    def add_sample(self, sample_x, sample_y):
        pass

    def train(self):
        pass

    def get_weights(self):
        pass

    def predict(self, input_x):
        pass

    def cost(self):
        pass

    def accuracy(self, test_set):
        pass