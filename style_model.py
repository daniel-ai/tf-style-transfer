import tensorflow as tf
import scipy.io
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

# change to your file locations
CONTENT = './image_input/golden_gate.jpg'
STYLE = './image_input/style.jpg'
SAVE = './image_output/'
VGGPATH = './models/imagenet-vgg-verydeep-16.mat'

### don't change anything below unless you understand

# description of pretrained neural network layers
VGG16DESC = 'conv2d-64-3-3-1-1, relu, conv2d-64-3-3-1-1, relu, mpool-2-2-2-2, conv2d-128-3-3-1-1, relu, conv2d-128-3-3-1-1, relu, mpool-2-2-2-2, conv2d-256-3-3-1-1, relu, conv2d-256-3-3-1-1, relu, conv2d-256-3-3-1-1, relu, mpool-2-2-2-2, conv2d-512-3-3-1-1, relu, conv2d-512-3-3-1-1, relu, conv2d-512-3-3-1-1, relu, mpool-2-2-2-2, conv2d-512-3-3-1-1, relu, conv2d-512-3-3-1-1, relu, conv2d-512-3-3-1-1, relu, mpool-2-2-2-2'
# weights on layers used to evaluate content/style transfer loss
C_WEIGHTS = dict([('layer_16',1)])
S_WEIGHTS = dict([('layer_4',1), ('layer_9',1), ('layer_16',1), ('layer_23',1)])
TV_WEIGHTS = 0.000001

def args():
    parser = ArgumentParser()
    parser.add_argument('--content', help='content image location', type=str, default=CONTENT)
    parser.add_argument('--style', help='style image location', type=str, default=STYLE)
    parser.add_argument('--save', help='output image save location', type=str, default=SAVE)
    parser.add_argument('--lr', help='learning rate', type=float, default=10)
    parser.add_argument('--iteration', help='total iterations for training ', type=int, default=500)
    parser.add_argument('--eval_iteration', help='iterations for evaluating loss and saving output image', type=int, default=100)
    options = parser.parse_args()
    return options

def main():
    global options
    options = args()
    g = tf.Graph()
    with g.as_default(), tf.Session() as sess:
        c_img = tf.read_file(options.content)
        c_img = tf.image.decode_jpeg(c_img)
        s_img = tf.read_file(options.style)
        s_img = tf.image.decode_jpeg(s_img)
        stylize(c_img, s_img, sess)

def stylize(c_img, s_img, sess):

    c_img = preprocess(sess.run(c_img), 'vgg')
    s_img = preprocess(sess.run(s_img), 'vgg')
    o_img = tf.Variable(c_img, name='stylized_img')
    sess.run(tf.initialize_variables([o_img]))
    weights = load_vgg16()
    vgg16 = Network(VGG16DESC, weights)
    c_net = vgg16.eval(c_img, C_WEIGHTS.keys())
    s_net = vgg16.eval(s_img, S_WEIGHTS.keys())
    o_net = vgg16.eval(o_img, set(C_WEIGHTS.keys()+S_WEIGHTS.keys()))

    c_loss = 0
    for layer in C_WEIGHTS.keys():
        _, h, w, c = c_net[layer].get_shape().as_list()
        c_loss += C_WEIGHTS[layer]*tf.nn.l2_loss(o_net[layer]-c_net[layer])/(h*w*c)
    s_loss = 0
    for layer in S_WEIGHTS.keys():
        _, h, w, c = s_net[layer].get_shape().as_list()
        s_loss += S_WEIGHTS[layer]*tf.nn.l2_loss(Gram(o_net[layer]) - Gram(s_net[layer]))
    _, h, w, c = c_img.get_shape().as_list()
    tv_loss = TV_WEIGHTS*tf.nn.l2_loss(o_img[:,1:,:,:] - o_img[:,:-1,:,:]) + tf.nn.l2_loss(o_img[:,:,1:,:] - o_img[:,:,:-1,:])/(h*w*c)
    loss = c_loss + s_loss + tv_loss
    train_step = tf.train.AdamOptimizer(options.lr).minimize(loss)

    time_start = datetime.now()
    sess.run(tf.initialize_all_variables())
    print 'Training started'
    for i in range(options.iteration):
        train_step.run()
        if  (i+1) % options.eval_iteration == 0:
            time_end = datetime.now()
            print 'iteration %d: c_loss = %.1f, s_loss = %.1f, tv_loss = %.1f, ave_time = %s' %(i+1, c_loss.eval(), s_loss.eval(), tv_loss.eval(), (time_end-time_start)/options.eval_iteration)
            time_start = datetime.now()
            save_img = Image.fromarray(deprocess(o_img.eval(), 'vgg'), "RGB")
            save_img.save(SAVE+'stylized_image_{}.jpg'.format(int((i+1)/options.eval_iteration)))
    print 'Training finished'

class Network:
    '''
    initiate neural network based on description string and evaluate content and style features for vgg/trained networks given input/conent/style images
    '''

    def __init__(self, modeldesc, initial=None, trainable=False):
        self.layers = [layer.strip().split('-') for layer in modeldesc.split(',')]
        self.initial = initial
        self.trainable = trainable

    def eval(self, i_img, layers):
        o_img = i_img
        net = {}
        for k, layer in enumerate(self.layers):
            with tf.variable_scope('layer_{}'.format(k+1)):
                if layer[0] == 'conv2d':
                    if self.initial is not None and not self.trainable:
                        kernel = tf.constant(self.initial[k][0], dtype=tf.float32)
                        bias = tf.constant(self.initial[k][1], dtype=tf.float32)
                    o_img = tf.nn.conv2d(o_img, kernel, [1, int(layer[4]), int(layer[5]), 1], 'SAME' )
                    o_img = tf.nn.bias_add(o_img, bias)
                elif layer[0] == 'relu':
                    o_img = tf.nn.relu(o_img)
                elif layer[0] == 'mpool':
                    o_img = tf.nn.max_pool(o_img, ksize=(1, int(layer[1]), int(layer[2]), 1), strides=(1, int(layer[3]), int(layer[4]), 1),padding='SAME')
                net['layer_{}'.format(k+1)] = o_img
        eval_layers = {layer:net[layer] for layer in layers}
        return eval_layers

def Gram(layer):
    _, h, w, c = layer.get_shape().as_list()
    flat = tf.reshape(layer, [-1, c])
    gram = tf.matmul(tf.transpose(flat), flat)/(h*w*c)
    return gram

def preprocess(image, method):
    if method == 'vgg':
        vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,3))
        image = image - vgg_mean
        return tf.constant(image[np.newaxis,:,:,:], dtype=tf.float32)

def deprocess(image, method):
    if method == 'vgg':
        vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,3))
        image = image[0] + vgg_mean
        return np.clip(image, 0, 255).astype(np.uint8)

def load_vgg16():
    vgg16_layers = scipy.io.loadmat(VGGPATH)['layers'][0]
    weights = []
    for k in range(len(vgg16_layers)):
        if vgg16_layers[k][0][0][1][0] == 'conv':
            kernel = np.transpose(vgg16_layers[k][0][0][2][0][0], (1,0,2,3))
            bias = np.reshape(vgg16_layers[k][0][0][2][0][1],-1)
            weights.append([kernel, bias])
        else:
            weights.append([])
    return weights

if __name__ == '__main__':
  main()