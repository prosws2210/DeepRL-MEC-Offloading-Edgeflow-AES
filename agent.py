from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf  # Use the compatibility module
tf.disable_v2_behavior()           # Disable TF2 behavior to run TF1 code

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.runtime_version')

class MemoryDNN:
    def __init__(self,
                 net,
                 learning_rate=0.01,
                 training_interval=10,
                 batch_size=100,
                 memory_size=1000,
                 output_graph=False):

        assert len(net) == 4, "Only 4-layer DNN supported: [input, hidden1, hidden2, output]"

        self.net = net
        self.lr = learning_rate
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.enumerate_actions = []
        self.memory_counter = 1
        self.cost_his = []

        # Initialize memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        tf.reset_default_graph()
        self._build_net()
        self.sess = tf.Session()

        # ADDED: Saver for saving the model
        self.saver = tf.train.Saver()

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        def build_layers(h, c_names, net, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [net[0], net[1]], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, net[1]], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(h, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [net[1], net[2]], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, net[2]], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            with tf.variable_scope('M'):
                w3 = tf.get_variable('w3', [net[2], net[3]], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, net[3]], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3

            return out

        self.h = tf.placeholder(tf.float32, [None, self.net[0]], name='h')
        self.m = tf.placeholder(tf.float32, [None, self.net[-1]], name='mode')

        with tf.variable_scope('memory_net'):
            c_names = ['memory_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0., 1 / self.net[0])
            b_initializer = tf.constant_initializer(0.1)
            self.m_pred = build_layers(self.h, c_names, self.net, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.m, logits=self.m_pred))

        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr, 0.09).minimize(self.loss)

    def remember(self, h, m):
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((h, m))
        self.memory_counter += 1

    def encode(self, h, m):
        self.remember(h, m)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        h_train = batch_memory[:, :self.net[0]]
        m_train = batch_memory[:, self.net[0]:]

        _, self.cost = self.sess.run([self._train_op, self.loss], feed_dict={self.h: h_train, self.m: m_train})
        assert self.cost >= 0
        self.cost_his.append(self.cost)

    def decode(self, h, k=1, mode='OP'):
        h = h[np.newaxis, :]
        m_pred = self.sess.run(self.m_pred, feed_dict={self.h: h})

        if mode == 'OP':
            return self.knm(m_pred[0], k)
        elif mode == 'KNN':
            return self.knn(m_pred[0], k)
        else:
            raise ValueError("Mode must be 'OP' or 'KNN'")

    def knm(self, m, k=1):
        m_list = [1 * (m > 0)]

        if k > 1:
            m_abs = abs(m)
            idx_list = np.argsort(m_abs)[:k - 1]
            for i in range(k - 1):
                if m[idx_list[i]] > 0:
                    m_list.append(1 * (m - m[idx_list[i]] > 0))
                else:
                    m_list.append(1 * (m - m[idx_list[i]] >= 0))
        return m_list

    def knn(self, m, k=1):
        if not self.enumerate_actions:
            import itertools
            self.enumerate_actions = np.array(list(map(list, itertools.product([0, 1], repeat=self.net[0]))))

        sqd = np.sum((self.enumerate_actions - m) ** 2, axis=1)
        idx = np.argsort(sqd)
        return self.enumerate_actions[idx[:k]]

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.xlabel('Time Frames')
        plt.ylabel('Training Loss')
        plt.title('DNN Training Loss Over Time')
        plt.grid(True)
        plt.show()

    # ADDED: Method to save the model
    def save_model(self, file_path):
        """Saves the trained model to the specified path."""
        self.saver.save(self.sess, file_path)
        print(f"Model saved in path: {file_path}")