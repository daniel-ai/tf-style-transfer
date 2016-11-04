"""
tensorflow implementation of the paper
Perceptual Losses for Real-Time Style Transfer and Super-Resolution
(style transfer part)
"""
import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime

# description of neural network layers in the form of
# 'layer_type-input_channels-output_channels-kernel_height-kernel_width
# -stride_height-stride-width'
# if the parameters are needed for the corresponding layer_type
# layer_type norm = instance normalization
# layer_type res = a residual block of [conv, norm, relu, conv, norm]
# layer_type tanh_resc = (tanh+1)*255/2
# only the first 23 layers of VGG 16 neural network are used
VGG16DESC = (
    'conv2d-3-3-3-64-1-1, relu, conv2d-3-3-64-64-1-1, relu, mpool-2-2-2-2,'
    'conv2d-3-3-64-128-1-1, relu, conv2d-3-3-128-128-1-1, relu, mpool-2-2-2-2,'
    'conv2d-3-3-128-256-1-1, relu, conv2d-3-3-256-256-1-1, relu,'
    'conv2d-3-3-256-256-1-1, relu, mpool-2-2-2-2, conv2d-3-3-256-512-1-1,'
    'relu, conv2d-3-3-512-512-1-1, relu, conv2d-3-3-512-512-1-1, relu')
MODELDESC = (
    'conv2d-9-9-3-32-1-1, norm-32, relu, conv2d-3-3-32-64-2-2, norm-64, relu,'
    'conv2d-3-3-64-128-2-2, norm-128, relu, res-3-3-128-128-1-1,'
    'res-3-3-128-128-1-1, res-3-3-128-128-1-1, res-3-3-128-128-1-1,'
    'res-3-3-128-128-1-1, tconv2d-3-3-128-64-2-2, norm-64, relu,'
    'tconv2d-3-3-64-32-2-2, norm-32, relu, tconv2d-9-9-32-3-1-1, tanh_resc')
# weights used to evaluate content and style loss
C_WEIGHTS = dict([('layer_16',1)])
S_WEIGHTS = dict([('layer_4',0.01), ('layer_9',0.01), ('layer_16',0.01),
                 ('layer_23',0.01)])
TV_WEIGHTS = 0.00001


def args():
    parser = ArgumentParser()
    parser.add_argument('--name', type=str, default='train',
                        help='name of the style transfer model')
    parser.add_argument('--content_dir', type=str, default='./image_input/',
                        help='location of images (c_*.jpg) to be stylized')
    parser.add_argument('--style', type=str, default='./image_input/style.jpg',
                        help='the styple image to be trained')
    parser.add_argument('--train_dir', type=str, default='./COCO/',
                        help='directory of content images for training')
    parser.add_argument('--image_size', nargs='+', type=int,
                        default=[256, 256], help='size of training images')
    parser.add_argument('--vggfile', type=str,
                        default='./models/imagenet-vgg-verydeep-16.mat',
                        help='location of pre-trained vgg model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='training batch size')
    parser.add_argument('--val_size', type=int, default=10,
                        help='number of batches used for validation')
    parser.add_argument('--global_step', type=int, default=0,
                        help='global count of training iterations')
    parser.add_argument('--max_iteration', type=int, default=20000,
                         help='maximum number of iterations for training')
    parser.add_argument('--eval_iteration', type=int, default=1000,
                        help='evaluate loss per number of iterations')
    parser.add_argument('--save_iteration', type=int, default=5000,
                         help='save session file per number of iterations',)
    parser.add_argument('--stylize', action='store_true',
                        help='bool whether to stylize or train models')
    parser.add_argument('--model', type=str, default='./models/model_single',
                        help='model file for save/restore')
    parser.add_argument('--output_dir', type=str, default='./image_output/',
                        help='path to save stylized images')
    options = parser.parse_args()
    return options


def main():
    """ train style transfer models or stylize images """
    options = args()
    with tf.Graph().as_default():
        if options.stylize:
            stylize(options.name, options.content_dir, options.model,
                    options.output_dir)
        else:
            train(options.name, options.style, options.train_dir,
                  options.image_size, options.global_step, options.lr,
                  options.batch_size, options.val_size, options.vggfile,
                  options.model, options.max_iteration,
                  options.eval_iteration, options.save_iteration)


def train(name, style, train_dir, image_size, global_step, lr, batch_size,
          val_size, vggfile, model, max_iteration, eval_iteration,
          save_iteration):
    """ train a single style tranfer neural network given a specific style

    see args help for definition

    """

    style_image = scipy.misc.imread(style)
    style_image = tf.to_float(style_image[np.newaxis,:,:,:])
    # create a tf placeholder for training content images
    input_shape = [batch_size] + image_size + [3]
    content_image = tf.placeholder(tf.float32, input_shape)
    style_net = Network(MODELDESC, name, trainable=True)
    output_image = style_net.image(content_image)

    c_loss, s_loss, tv_loss = loss_vgg(
        style_image, content_image, output_image, vggfile)
    loss = c_loss + s_loss + tv_loss
    # use Adam optimizer with gradient clipping
    optimizer = tf.train.AdamOptimizer(lr)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 1)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    train_images = train_set(train_dir)
    # first val_size of batches are used as the validation set
    val_set = [train_images(batch_size) for k in range(val_size)]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if global_step == 0:
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, model+'-{}'.format(global_step))
        train_images.index = global_step % (
            train_images.number - batch_size*val_size) + batch_size*val_size
        print 'Started training with {} trainable tensors:'.format(
            len(tf.trainable_variables()))
        sys.stdout.flush()
        time_start = datetime.now()
        for i in range(max_iteration):
            if train_images.index+batch_size > train_images.number:
                train_images.index = batch_size*val_size
            image = train_images(batch_size)
            sess.run(train_op, {content_image: image})
            global_step += batch_size
            if  (i+1) % eval_iteration == 0 or (i+1) == max_iteration:
                time_end = datetime.now()
                if (i+1) % eval_iteration == 0:
                    iterations = eval_iteration
                else:
                    iterations = (i+1) % eval_iteration
                val_losses = []
                for val_image in val_set:
                    val_losses.append(
                        sess.run([c_loss, s_loss, tv_loss],
                                 {content_image: val_image}))
                val_losses = np.stack(val_losses)
                [val_c_loss, val_s_loss, val_tv_loss] = np.mean(val_losses,
                    axis=0)/batch_size
                print ('iteration %d: c_loss = %.1f, s_loss = %.1f, tv_loss = '
                    '%.1f, ave_time = %s') %(
                    i+1, val_c_loss, val_s_loss, val_tv_loss,
                    (time_end-time_start)/iterations)
                sys.stdout.flush()
                time_start = datetime.now()
            if  (i+1) % save_iteration == 0 or (i+1) == max_iteration:
                saver.save(sess, model, global_step, write_meta_graph=False)
                print 'Session file {}-{} saved.'.format(model, global_step)
                sys.stdout.flush()
        print 'Training finished (global step = {}).'.format(global_step)


def stylize(name, content_dir, model, output_dir):
    """ stylize images using pre-trained nerual network

    see args help for definition

    """
    content_images = {}
    for root, dirs, files in os.walk(content_dir):
        for filenm in files:
            if filenm[0:2] == 'c_':
                content_image = scipy.misc.imread(os.path.join(root, filenm))
                c_name = filenm[2:-4]
                content_images[c_name] = tf.to_float(
                    content_image[np.newaxis,:,:,:])
    print 'Stylizing {} content images'.format(len(content_images))
    style_net = Network(MODELDESC, name, trainable=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model)
        for c_name in content_images.keys():
            output_image = style_net.image(content_images[c_name])
            output_image = Image.fromarray(
                output_image.eval()[0].astype(np.uint8), "RGB")
            output_image.save(output_dir+'stylized_{}.jpg'.format(c_name))
    print '{} stylized image saved to {}.'.format(
        len(content_images), output_dir)


class train_set:
    """ training data feeder """
    def __init__(self, train_dir):
        self.files = []
        for root, dirs, files in os.walk(train_dir):
            for filenm in files:
                if filenm[-4:] == '.jpg':
                    self.files.append(os.path.join(root,filenm))
        self.number = len(files)
        self.index = 0

    def __call__(self, batch_size):
        if self.index+batch_size > self.number:
            raise ValueError("Training set exhausted")
        batch_files = self.files[self.index:self.index+batch_size]
        self.index += batch_size
        images = [scipy.misc.imread(file) for file in batch_files]
        images = np.stack(images, axis=0)
        return images.astype(np.float32)


def loss_vgg(style_image, content_image, output_image, vggfile):
    """ calculate losses based on pre-trained vgg neural network """
    c_layers = C_WEIGHTS.keys()
    s_layers = S_WEIGHTS.keys()
    vgg16_filters = load_vgg(vggfile)

    vgg16 = Network(VGG16DESC, 'vgg16', initial=vgg16_filters, process=True)
    s_net = vgg16.layers(style_image, s_layers)
    c_net = vgg16.layers(content_image, c_layers)
    o_net = vgg16.layers(output_image, set(c_layers+s_layers))

    c_loss = 0.0
    for layer in c_layers:
        _, h, w, c = c_net[layer].get_shape().as_list()
        c_loss += C_WEIGHTS[layer]*tf.nn.l2_loss(
            o_net[layer]-c_net[layer])/(h*w*c)
    s_loss = 0.0
    for layer in s_layers:
        bs, _, _, c = o_net[layer].get_shape().as_list()
        s_loss += S_WEIGHTS[layer]*tf.nn.l2_loss(
            Gram(o_net[layer], bs) - Gram(s_net[layer], bs))
    tv_loss = TV_WEIGHTS*(
        tf.nn.l2_loss(output_image[:,1:,:,:] - output_image[:,:-1,:,:])
        + tf.nn.l2_loss(output_image[:,:,1:,:] - output_image[:,:,:-1,:]))
    return c_loss, s_loss, tv_loss


class Network(object):
    """ a neural network class generated from descriptions string
    1. load the parameters from pre-trained neural network and evaluate layers
       given the input image
    2. train new neural network by initializing the parameters and feeding
       the training data
    """
    def __init__(self, modeldesc, name, initial=None, trainable=False,
                 process=False):
        self.name = name
        self.layers_desc = []
        for layer in modeldesc.split(','):
            self.layers_desc.append(layer.strip().split('-'))
        self.n_layers = len(self.layers_desc)
        self.trainable = trainable
        self.process = process
        self.filters = {}
        for k, layer_desc in enumerate(self.layers_desc):
            layer_name = 'layer_{}'.format(k+1)
            layer_initial = initial[layer_name] if initial else None
            with tf.variable_scope('{}_{}'.format(name, layer_name)):
                if layer_desc[0] == 'res':
                    self.filters[layer_name] = res_filter_init(
                        layer_desc, layer_initial, trainable)
                else:
                    self.filters[layer_name] = basic_filter_init(
                        layer_desc, layer_initial, trainable)

    def layers(self, image, layers):
        """ evaluate certain vgg layers' values given an image """
        output_image = preprocess(image) if self.process else image
        eval_layers = {}
        max_layer = max([int(layer[6:]) for layer in layers])
        for k in range(max_layer):
            layer_name = 'layer_{}'.format(k+1)
            if self.layers_desc[k][0] == 'res':
                output_image = res_layer_eval(
                    output_image, self.layers_desc[k],
                    self.filters[layer_name])
            else:
                output_image = basic_layer_eval(
                    output_image, self.layers_desc[k],
                    self.filters[layer_name])
            if layer_name in layers:
                eval_layers[layer_name] = output_image
        return eval_layers

    def image(self, image):
        """ generate the final stylized image """
        last_layer_name = 'layer_{}'.format(self.n_layers)
        output_image = self.layers(image, [last_layer_name])[last_layer_name]
        return output_image


def res_layer_eval(image, layer_desc, layer_filter):
    """ evaluate one block of residual network """
    conv_layer_desc = ['conv2d'] + layer_desc[1:]
    output_image = basic_layer_eval(image, conv_layer_desc, layer_filter[0:2])
    output_image = basic_layer_eval(output_image, ['norm'], layer_filter[2:4])
    output_image = basic_layer_eval(output_image, ['relu'], [])
    output_image = basic_layer_eval(
        output_image, conv_layer_desc, layer_filter[4:6])
    output_image = basic_layer_eval(output_image, ['norm'], layer_filter[6:8])
    return image + output_image


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
        mean, variance = tf.nn.moments(image, [1,2], keep_dims=True)
        _, _, _, c = image.get_shape().as_list()
        scales = tf.reshape(layer_filter[0], [1,1,1,c])
        offsets = tf.reshape(layer_filter[1], [1,1,1,c])
        return tf.nn.batch_normalization(
            image, mean, variance, offsets, scales, 1e-5)
    elif layer_desc[0] == 'tconv2d':
        b, h, w, c = image.get_shape().as_list()
        s_1, s_2 = map(int, layer_desc[5:7])
        image = tf.nn.conv2d_transpose(
            image, layer_filter[0],
            [b,s_1*h,s_2*w,int(layer_desc[4])], [1,s_1,s_2,1], 'SAME')
        return tf.nn.bias_add(image, layer_filter[1])
    elif layer_desc[0] == 'tanh_resc':
        return (tf.tanh(image)+1)*255/2
    elif layer_desc[0] == 'sigmoid_res':
        return (tf.sigmoid(image))*255


def res_filter_init(layer_desc, layer_initial, trainable):
    """ initialize parameters for one block of residual network """
    conv_layer_desc = ['conv2d'] + layer_desc[1:]
    norm_layer_desc = ['norm', layer_desc[3]]
    with tf.variable_scope('conv_1'):
        c1_layer_initial = layer_initial[0:2] if layer_initial else None
        conv_1 = basic_filter_init(
            conv_layer_desc, c1_layer_initial, trainable)
    with tf.variable_scope('norm_1'):
        n1_layer_initial = layer_initial[2:4] if layer_initial else None
        norm_1 = basic_filter_init(
            norm_layer_desc, n1_layer_initial, trainable)
    with tf.variable_scope('conv_2'):
        c2_layer_initial = layer_initial[4:6] if layer_initial else None
        conv_2 = basic_filter_init(
            conv_layer_desc, c2_layer_initial, trainable)
    with tf.variable_scope('norm_2'):
        n2_layer_initial = layer_initial[6:8] if layer_initial else None
        norm_2 = basic_filter_init(
            norm_layer_desc, n2_layer_initial, trainable)
    return conv_1 + norm_1 + conv_2 + norm_2


def basic_filter_init(layer_desc, layer_initial, trainable):
    """ initialize parameters for basic types of neural network layers """
    if layer_desc[0] in ['relu', 'mpool', 'tanh_resc', 'sigmoid_res']:
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
                    initializer=tf.constant_initializer(1)),
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


def Gram(layer, batch_size):
    """ calcualte the Gram matrix for the loss function """
    if type(layer).__module__ == 'numpy':
        bs, h, w, c = layer.shape
    else:
        bs, h, w, c = layer.get_shape().as_list()
    if bs ==  1 and batch_size > 1:
        rep_layer = tf.pack([layer for k in range(batch_size)])
        flat = tf.reshape(rep_layer, [batch_size, -1, c])
    else:
        flat = tf.reshape(layer, [bs, -1, c])
    return tf.batch_matmul(tf.transpose(flat, [0,2,1]), flat)/(h*w*c)


def preprocess(image):
    """ preprocess images for VGG neural network """
    vgg_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return image-vgg_mean


def load_vgg(file):
    """ load pre-trained vgg nerual networks """
    vgg_layers = scipy.io.loadmat(file)['layers'][0]
    filters = {}
    for k in range(len(vgg_layers)):
        if vgg_layers[k][0][0][1][0] == 'conv':
            weights = np.array(vgg_layers[k][0][0][2][0][0])
            biases = np.reshape(vgg_layers[k][0][0][2][0][1], -1)
            filters['layer_{}'.format(k+1)] = [weights, biases]
        else:
            filters['layer_{}'.format(k+1)] = []
    return filters


def images_resize(image_size, train_dir):
    """ resize training images """
    for root, dirs, files in os.walk(train_dir):
        for filenm in files:
            image = scipy.misc.imread(os.path.join(root,filenm))
            if len(image.shape) != 3 or image.shape[-1] != 3:
                os.remove(os.path.join(root,filenm))
            if image.shape[0]/image.shape[1] >= image_size[0]/image_size[1]:
                image = scipy.misc.imresize(
                    image, [int(image_size[0]*image.shape[0]/image.shape[1]),
                    image_size[1]])
                start_index = int((image.shape[0]-image_size[0])/2)
                image = image[start_index:start_index+image_size[0],:]
            else:
                image = scipy.misc.imresize(
                    image, [image_size[0],
                    int(image_size[1]*image.shape[1]/image.shape[0])])
                start_index = int((image.shape[1]-image_size[1])/2)
                image = image[:,start_index:start_index+image_size[1]]
            os.remove(os.path.join(root,filenm))
            Image.fromarray(image).save(os.path.join(root,filenm))


if __name__ == '__main__':
  main()



