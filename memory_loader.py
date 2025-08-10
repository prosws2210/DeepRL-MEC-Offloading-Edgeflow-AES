# memory_loader.py
# This class is specifically designed to load the .ckpt model trained by new_memory.py

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() # Must disable V2 behavior to run TF1 code

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.runtime_version')

class MemoryDNN_TF1_Loader:
    def __init__(self, net):
        """
        Initializes the loader.
        'net' must be the same architecture as the trained model.
        e.g., [10, 120, 80, 10] for 10 users.
        """
        assert len(net) == 4, "Net must be a 4-layer list: [input, h1, h2, output]"
        self.net = net
        tf.reset_default_graph()
        # Build the exact same graph as the training script
        self._build_net()
        self.sess = tf.Session()
        # The saver is used to restore the variables
        self.saver = tf.train.Saver()

    def _build_net(self):
        """
        This method builds a TensorFlow graph that is IDENTICAL to the one
        in new_memory.py. Every name and scope must match exactly.
        """
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

        # We only need the input placeholder for prediction
        self.h = tf.placeholder(tf.float32, [None, self.net[0]], name='h')

        # The variable scope here MUST be 'memory_net' to match the saved model
        with tf.variable_scope('memory_net'):
            c_names = ['memory_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            # The initializers must be defined, even though we are loading trained values over them
            w_initializer = tf.random_normal_initializer(0., 1 / self.net[0])
            b_initializer = tf.constant_initializer(0.1)
            self.m_pred = build_layers(self.h, c_names, self.net, w_initializer, b_initializer)

    def load_model(self, model_path):
        """Loads a saved .ckpt model from the specified path prefix."""
        try:
            self.saver.restore(self.sess, model_path)
            print(f"✅ TF1 .ckpt model restored successfully from: {model_path}")
        except Exception as e:
            print(f"🚨🚨 An error occurred while restoring the TF1 model: {e} 🚨🚨")
            raise # Re-raise the exception to stop the script

    def decode(self, h):
        """Predicts offloading decisions using the loaded model."""
        h = h[np.newaxis, :]
        # self.m_pred is the raw output (logits) of the network
        m_logits = self.sess.run(self.m_pred, feed_dict={self.h: h})
        # A positive logit corresponds to a probability > 0.5
        return (m_logits[0] > 0).astype(int)