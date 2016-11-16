import tensorflow as tf
import numpy as np


class Network:
    """ a neural network class generated from descriptions string
    1. load the parameters from pre-trained neural network and evaluate layers
       given the input image
    2. train new neural network by initializing the parameters and feeding
       the training data
    """
    def __init__(self, modeldesc, name, styles=None, initial=None,
                 trainable=False, process=False):
        self.layers_desc = []
        for layer in modeldesc.split(','):
            self.layers_desc.append(layer.strip().split('-'))
        self.n_layers = len(self.layers_desc)
        self.styles = styles
        self.process = process
        self.trainable = trainable
        self.filters = {}
        for k, layer_desc in enumerate(self.layers_desc):
            layer_name = 'layer_{}'.format(k+1)
            layer_initial = initial[layer_name] if initial else None
            with tf.variable_scope('{}_{}'.format(name, layer_name)):
                if layer_desc[0] == 'res':
                    self.filters[layer_name] = res_filter_init(
                        layer_desc, styles, layer_initial, trainable)
                elif layer_desc[0] == 's_norm':
                    self.filters[layer_name] = s_norm_filter_init(
                        layer_desc, styles, layer_initial, trainable)
                else:
                    self.filters[layer_name] = basic_filter_init(
                        layer_desc, layer_initial, trainable)

    def layers(self, image, layers, style=None):
        """ evaluate certain vgg layers' values given an image """
        output_image = preprocess(image) if self.process else image
        eval_layers = {}
        max_layer = max([int(layer[6:]) for layer in layers])
        for k in range(max_layer):
            layer_name = 'layer_{}'.format(k+1)
            if self.layers_desc[k][0] == 'res':
                output_image = res_layer_eval(
                    output_image, self.layers_desc[k],
                    self.filters[layer_name], style)
            elif self.layers_desc[k][0] == 's_norm':
                output_image = s_norm_layer_eval(
                    output_image, self.layers_desc[k],
                    self.filters[layer_name], style)
            else:
                output_image = basic_layer_eval(
                    output_image, self.layers_desc[k],
                    self.filters[layer_name])
            if layer_name in layers:
                eval_layers[layer_name] = output_image
        return eval_layers

    def images(self, image, styles):
        """ generate the final stylized image """
        last_layer_name = 'layer_{}'.format(self.n_layers)
        output_images = {}
        for style in styles:
            output_images[style] = self.layers(
                image, [last_layer_name], style)[last_layer_name]
        return output_images


def res_layer_eval(image, layer_desc, layer_filter, style):
    """ evaluate one block of residual network """
    conv_layer_desc = ['conv2d'] + layer_desc[1:]
    norm_layer_desc = ['s_norm', layer_desc[3]]
    output_image = basic_layer_eval(image, conv_layer_desc, layer_filter[0])
    output_image = s_norm_layer_eval(
        output_image, norm_layer_desc, layer_filter[1], style)
    output_image = basic_layer_eval(output_image, ['relu'], [])
    output_image = basic_layer_eval(
        output_image, conv_layer_desc, layer_filter[2])
    output_image = s_norm_layer_eval(
        output_image, norm_layer_desc, layer_filter[3], style)
    return image + output_image


def s_norm_layer_eval(image, layer_desc, layer_filter, style):
    """ evaluate one layer of stylized norm """
    norm_layer_desc = ['norm', layer_desc[1]]
    return basic_layer_eval(image, norm_layer_desc, layer_filter[style])


def basic_layer_eval(image, layer_desc, layer_filter):
    """ evaluate one layer of basic types of neural network """
    if layer_desc[0] == 'conv2d':
        kernel = map(int,layer_desc[1:3])
        stride = map(int,layer_desc[5:7])
        image = padding(image, kernel, stride)
        image = tf.nn.conv2d(image, layer_filter[0], [1]+stride+[1], 'VALID')
        return tf.nn.bias_add(image, layer_filter[1])
    elif layer_desc[0] == 'relu':
        return tf.nn.relu(image)
    elif layer_desc[0] == 'mpool':
        kernel = map(int,layer_desc[1:3])
        stride = map(int,layer_desc[3:5])
        image = padding(image, kernel, stride)
        return tf.nn.max_pool(image, [1]+kernel+[1], [1]+stride+[1], 'VALID')
    elif layer_desc[0] == 'norm':
        mean, var = tf.nn.moments(image, [1,2], keep_dims=True)
        var = tf.maximum(var, 0.)
        _, _, _, c = image.get_shape().as_list()
        scales = tf.reshape(layer_filter[0], [1,1,1,c])
        offsets = tf.reshape(layer_filter[1], [1,1,1,c])
        return tf.nn.batch_normalization(
            image, mean, var, offsets, scales, 1e-5)
    elif layer_desc[0] == 'tconv2d':
        b, h, w, c = image.get_shape().as_list()
        s_1, s_2 = map(int, layer_desc[5:7])
        image = tf.nn.conv2d_transpose(
            image, layer_filter[0],
            [b,s_1*h,s_2*w,int(layer_desc[4])], [1,s_1,s_2,1], 'SAME')
        return tf.nn.bias_add(image, layer_filter[1])
    elif layer_desc[0] == 'tanh_resc':
        return (tf.tanh(image)+1)*255/2
    elif layer_desc[0] == 'sigmoid_resc':
        return (tf.sigmoid(image))*255


def res_filter_init(layer_desc, styles, layer_initial, trainable):
    """ initialize parameters for one block of residual network """
    conv_layer_desc = ['conv2d'] + layer_desc[1:]
    norm_layer_desc = ['s_norm', layer_desc[3]]
    with tf.variable_scope('conv_1'):
        c1_layer_initial = layer_initial[0] if layer_initial else None
        conv_1 = basic_filter_init(
            conv_layer_desc, c1_layer_initial, trainable)
    with tf.variable_scope('s_norm_1'):
        n1_layer_initial = layer_initial[1] if layer_initial else None
        norm_1 = s_norm_filter_init(
            norm_layer_desc, styles, n1_layer_initial, trainable)
    with tf.variable_scope('conv_2'):
        c2_layer_initial = layer_initial[2] if layer_initial else None
        conv_2 = basic_filter_init(
            conv_layer_desc, c2_layer_initial, trainable)
    with tf.variable_scope('s_norm_2'):
        n2_layer_initial = layer_initial[3] if layer_initial else None
        norm_2 = s_norm_filter_init(
            norm_layer_desc, styles, n2_layer_initial, trainable)
    return [conv_1, norm_1, conv_2, norm_2]


def s_norm_filter_init(layer_desc, styles, layer_initial, trainable):
    """ initialize parameters for stylized norm layer """
    layer_by_style = {}
    norm_layer_desc = ['norm', layer_desc[1]]
    for style in styles:
        norm_layer_initial = layer_initial[style] if layer_initial else None
        with tf.variable_scope(style):
            layer_by_style[style] = basic_filter_init(
                norm_layer_desc, norm_layer_initial, trainable)
    return layer_by_style


def basic_filter_init(layer_desc, layer_initial, trainable):
    """ initialize parameters for basic types of neural network layers """
    if layer_desc[0] in ['relu', 'mpool', 'tanh_resc', 'sigmoid_resc']:
        return []
    else:
        if layer_desc[0] in ['conv2d', 'tconv2d']:
            var_names = ['weights', 'biases']
        elif layer_desc[0] == 'norm':
            var_names = ['scales', 'offsets']
        if not trainable and layer_initial is not None:
            para_0 = tf.constant(
                layer_initial[0], name=var_names[0], dtype=tf.float32)
            para_1 = tf.constant(
                layer_initial[1], name=var_names[1], dtype=tf.float32)
        elif trainable and layer_initial is not None:
            para_0 = tf.Variable(
                layer_initial[0], name=var_names[0], dtype=tf.float32)
            para_1 = tf.Variable(
                layer_initial[1], name=var_names[1], dtype=tf.float32)
        elif trainable and layer_initial is None:
            if layer_desc[0] == 'conv2d':
                para_0 = tf.get_variable(
                    'weights', map(int, layer_desc[1:5]),
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                para_1 = tf.get_variable(
                    'biases', [int(layer_desc[4])],
                    initializer=tf.constant_initializer(0))
            elif layer_desc[0] == 'tconv2d':
                h, w, i_c, o_c = map(int, layer_desc[1:5])
                para_0 = tf.get_variable(
                    'weights', [h, w, o_c, i_c],
                    initializer=tf.truncated_normal_initializer(stddev=0.01))
                para_1 = tf.get_variable(
                    'biases', [o_c], initializer=tf.constant_initializer(0))
            elif layer_desc[0] == 'norm':
                para_0 = tf.get_variable(
                    'scales', int(layer_desc[1]),
                    initializer=tf.constant_initializer(1))
                para_1 = tf.get_variable(
                    'offsets', int(layer_desc[1]),
                    initializer=tf.constant_initializer(0))
        else:
            raise ValueError("Please load trained models for initialization.")
        return [para_0, para_1]


def padding(image, kernel, stride):
    """ apply reflect padding before convolutional layers"""
    bs, h, w, c = image.get_shape().as_list()
    pad_h = kernel[0] + (np.ceil(float(h)/float(stride[0]))-1)*int(stride[0])-h
    pad_h = max(0, pad_h)
    pad_w = kernel[1] + (np.ceil(float(w)/float(stride[1]))-1)*int(stride[1])-w
    pad_w = max(0, pad_w)
    paddings = [[0,0], [int(pad_h/2),int(pad_h)-int(pad_h/2)], [int(pad_w/2),
                int(pad_w)-int(pad_w/2)], [0,0]]
    return tf.pad(image, paddings, 'REFLECT')


def preprocess(image):
    """ preprocess images for VGG neural network """
    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return image-vgg_mean




