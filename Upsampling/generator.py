import tensorflow as tf
from Common import ops as ops
from math import log
class Generator(object):
    def __init__(self, opts,is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.up_ratio = self.opts.up_ratio
        


    def __call__(self, inputs, up_ratio=4):
        with tf.variable_scope(self.name, reuse=self.reuse):

            features = ops.feature_extraction(inputs, scope='feature_extraction', is_training=self.is_training, bn_decay=None)#[B,N,1,c]
            encode_ = ops.conv2d(features, 3, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=False, is_training=self.is_training,
                                       scope='fc_layer0', bn_decay=None,
                                       activation_fn=None, weight_decay=0.0)

            encode = tf.squeeze(encode_, [2])
            
            decode_all = []
            up_feature = features
            up_points = encode
            for ii in range(int(log(up_ratio,2))):
                up_feature, up_points = ops.unet_model(up_feature, up_points, is_training=self.is_training, scope = 'up_net')

                up_points = tf.squeeze(up_points, [2])
                decode_all.append(up_points)

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return encode, decode_all[0], up_points



