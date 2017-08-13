import tensorflow as tf
import tensorflow.contrib.slim as slim

def vgg_arg_scope():
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        padding="SAME") as arg_sc:
        return arg_sc


def vgg_19(inputs, scope="vgg_19"):
    with tf.variable_scope(scope, "vgg_19", [inputs]) as sc:
        end_points_collection = sc.name + "_end_points"
        with slim.arg_scope([slim.conv2d, slim.avg_pool2d],
                        outputs_collections=end_points_collection):
            with slim.arg_scope([slim.conv2d], trainable=False):
                net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope="conv1")
                net = slim.avg_pool2d(net, [2, 2], scope="pool1")
                net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope="conv2")
                net = slim.avg_pool2d(net, [2, 2], scope="pool2")
                net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope="conv3")
                net = slim.avg_pool2d(net, [2, 2], scope="pool3")
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope="conv4")
                net = slim.avg_pool2d(net, [2, 2], scope="pool4")
                net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope="conv5")
                net = slim.avg_pool2d(net, [2, 2], scope="pool5")
            
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        return net, end_points