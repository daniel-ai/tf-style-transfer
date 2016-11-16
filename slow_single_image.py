"""
tensorflow implementation of the paper
Image Style Transfer Using Convolutional Neural Networks
"""
import tensorflow as tf
import numpy as np
import scipy.io
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime

# description of neural network layers in the form of
# 'layer_type-kernel_height-kernel_width-input_channels-output_channels
#  -stride_height-stride-width'
# if the parameters are needed for the corresponding layer_type
# only the first 23 layers of VGG 16 neural network are used
VGG16DESC = (
    'conv2d-3-3-3-64-1-1, relu, conv2d-3-3-64-64-1-1, relu, mpool-2-2-2-2,'
    'conv2d-3-3-64-128-1-1, relu, conv2d-3-3-128-128-1-1, relu, mpool-2-2-2-2,'
    'conv2d-3-3-128-256-1-1, relu, conv2d-3-3-256-256-1-1, relu,'
    'conv2d-3-3-256-256-1-1, relu, mpool-2-2-2-2, conv2d-3-3-256-512-1-1,'
    'relu, conv2d-3-3-512-512-1-1, relu, conv2d-3-3-512-512-1-1, relu')
# weights used to evaluate content and style loss
C_WEIGHTS = dict([('layer_16',1)])
S_WEIGHTS = dict([('layer_4',0.001), ('layer_9',0.001), ('layer_16',0.001),
                 ('layer_23',0.001)])
TV_WEIGHTS = 0.00001


def args():
    parser = ArgumentParser()
    parser.add_argument('--content', help='content image', type=str,
                        default='./image_input/c_golden_gate.jpg')
    parser.add_argument('--style', help='style image', type=str,
                        default='./image_input/style.jpg')
    parser.add_argument('--vggfile', help='locaiton of pre-trained vgg model',
                        type=str,
                        default='./models/imagenet-vgg-verydeep-16.mat')
    parser.add_argument('--save_dir', help='path to save output images',
                        type=str, default='./image_output/')
    parser.add_argument('--lr', help='learning rate', type=float, default=10)
    parser.add_argument('--max_iteration',
                        help='total number of iterations for training ',
                        type=int, default=500)
    parser.add_argument('--eval_iteration',
                        help=('evaluate loss and save the output image '
                              'per number of iterations'),
                        type=int, default=100)
    options = parser.parse_args()
    return options


def main():
    options = args()
    with tf.Graph().as_default():
        c_img = scipy.misc.imread(options.content)
        c_img = tf.to_float(c_img[np.newaxis,:,:,:])
        s_img = scipy.misc.imread(options.style)
        s_img = tf.to_float(s_img[np.newaxis,:,:,:])
        # output image initialized as the original content image
        o_img = tf.Variable(c_img, name='stylized_image')

        vgg16_filters = load_vgg16(options.vggfile)
        vgg16 = Network(VGG16DESC, vgg16_filters)
        c_net = vgg16.eval(c_img, C_WEIGHTS.keys())
        s_net = vgg16.eval(s_img, S_WEIGHTS.keys())
        o_net = vgg16.eval(o_img, set(C_WEIGHTS.keys()+S_WEIGHTS.keys()))

        # the loss function consists of three parts
        # content loss - content vs output image on certian layers of vgg
        # style loss - style vs output image on certian layers of vgg
        # total variation loss of output image
        c_loss = 0.0
        for layer in C_WEIGHTS.keys():
            _, h, w, c = c_net[layer].get_shape().as_list()
            c_loss += C_WEIGHTS[layer]*tf.nn.l2_loss(
                o_net[layer]-c_net[layer])/(h*w*c)
        s_loss = 0.0
        for layer in S_WEIGHTS.keys():
            s_loss += S_WEIGHTS[layer]*tf.nn.l2_loss(
                Gram(o_net[layer]) - Gram(s_net[layer]))
        tv_loss = TV_WEIGHTS*(
            tf.nn.l2_loss(o_img[:,1:,:,:] - o_img[:,:-1,:,:])
            + tf.nn.l2_loss(o_img[:,:,1:,:] - o_img[:,:,:-1,:]))
        loss = c_loss + s_loss + tv_loss
        train_step = tf.train.AdamOptimizer(options.lr).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            print 'Training started.'
            time_start = datetime.now()
            for i in range(options.max_iteration):
                train_step.run()
                if  (i+1) % options.eval_iteration == 0:
                    time_end = datetime.now()
                    print ('iteration %d: c_loss = %.1f, s_loss = %.1f, '
                        'tv_loss = %.1f, ave_time = %s') %(
                        i+1, c_loss.eval(), s_loss.eval(), tv_loss.eval(),
                        (time_end-time_start)/options.eval_iteration)
                    time_start = datetime.now()
                    save_img = np.clip(
                        o_img.eval()[0], 0, 255).astype(np.uint8)
                    save_img = Image.fromarray(save_img, "RGB")
                    save_img.save(
                        options.save_dir+'stylized_image_{}.jpg'.format(
                            int((i+1)/options.eval_iteration)))
            print 'Training finished.'


class Network:
    """ initiate neural network based on description string and pre-trained
        models, and evaluate layers given conent/style/output images

    Args:
        modeldesc: a string description of the neural network layers
        filters: a dictionary {layer_num:[list of parameters]} of pre-trained models
        process: bool, whether to  use preprocesss on input image
    Returns:
        a network class with method eval()
    """
    def __init__(self, modeldesc, filters, process=True):
        self.layers_desc = []
        for layer in modeldesc.split(','):
            self.layers_desc.append(layer.strip().split('-'))
        self.process = process
        self.filters = {}
        for k, layer in enumerate(self.layers_desc):
            layer_name = 'layer_{}'.format(k+1)
            if layer[0] == 'conv2d':
                weights = tf.constant(filters[k][0], dtype=tf.float32)
                biases = tf.constant(filters[k][1], dtype=tf.float32)
                self.filters[layer_name] = [weights, biases]
            else:
                self.filters[layer_name] = []

    def eval(self, i_img, layers):
        """ evalute vgg layers given the input image

        Args:
            i_img: input image
            layers: a list ['layer_num'] to be evaluated
        Returns:
            a dictionary {layer_num:layer_value}
        """
        o_img = preprocess(i_img) if self.process else i_img
        layers_eval = {}
        max_layer = max([int(layer[6:]) for layer in layers])
        for k in range(max_layer):
            layer_name = 'layer_{}'.format(k+1)
            layer_desc = self.layers_desc[k]
            if layer_desc[0] == 'conv2d':
                o_img = tf.nn.conv2d(
                    o_img, self.filters[layer_name][0],
                    [1,int(layer_desc[5]),int(layer_desc[6]),1], 'SAME' )
                o_img = tf.nn.bias_add(o_img, self.filters[layer_name][1])
            elif layer_desc[0] == 'relu':
                o_img = tf.nn.relu(o_img)
            elif layer_desc[0] == 'mpool':
                    o_img = tf.nn.max_pool(
                        o_img, (1,int(layer_desc[1]),int(layer_desc[2]),1),
                        (1,int(layer_desc[3]),int(layer_desc[4]),1), 'SAME')
            if layer_name in layers:
                layers_eval[layer_name] = o_img
        return layers_eval


def Gram(layer):
    """ calcuate the Gram matrix (see paper for definition)"""
    _, h, w, c = layer.get_shape().as_list()
    flat = tf.reshape(layer, [-1, c])
    gram = tf.matmul(tf.transpose(flat), flat)/(h*w*c)
    return gram


def preprocess(image):
    """ preprocess image for vgg neural network """
    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return image - vgg_mean


def load_vgg16(vggfile):
    """ load pre-trained vgg neural network parameters """
    vgg16_layers = scipy.io.loadmat(vggfile)['layers'][0]
    weights = []
    for k in range(len(vgg16_layers)):
        if vgg16_layers[k][0][0][1][0] == 'conv':
            kernel = np.array(vgg16_layers[k][0][0][2][0][0])
            bias = np.reshape(vgg16_layers[k][0][0][2][0][1],-1)
            weights.append([kernel, bias])
        else:
            weights.append([])
    return weights

if __name__ == '__main__':
  main()
