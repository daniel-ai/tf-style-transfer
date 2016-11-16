import tensorflow as tf
import numpy as np
import scipy.io
import nn_build

# description of neural network layers in the form of
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


def loss_vgg(style_images, content_image, output_images, vggfile):
    """ calculate losses based on pre-trained vgg neural network
        content loss - content vs output image on certian layers of vgg
        style loss - style vs output image on certian layers of vgg
        tv loss - total variation loss of output image
    """
    c_layers = C_WEIGHTS.keys()
    s_layers = S_WEIGHTS.keys()
    vgg16_filters = load_vgg(vggfile)
    vgg16 = nn_build.Network(
        VGG16DESC, 'vgg16', initial=vgg16_filters, process=True)
    c_net = vgg16.layers(content_image, c_layers)

    c_loss = 0.
    s_loss = 0.
    tv_loss = 0.
    for style in style_images:
        s_net = vgg16.layers(style_images[style], s_layers)
        o_net = vgg16.layers(output_images[style], set(c_layers+s_layers))
        for layer in c_layers:
            _, h, w, c = c_net[layer].get_shape().as_list()
            c_loss += C_WEIGHTS[layer]*tf.nn.l2_loss(
                o_net[layer]-c_net[layer])/(h*w*c)
        for layer in s_layers:
            bs, _, _, c = o_net[layer].get_shape().as_list()
            s_loss += S_WEIGHTS[layer]*tf.nn.l2_loss(
                Gram(o_net[layer], bs) - Gram(s_net[layer], bs))
        tv_loss += TV_WEIGHTS*(
            tf.nn.l2_loss(output_images[style][:,1:,:,:]
                          - output_images[style][:,:-1,:,:])
            + tf.nn.l2_loss(output_images[style][:,:,1:,:]
                            - output_images[style][:,:,:-1,:]))
    style_num = len(style_images)
    return c_loss/style_num, s_loss/style_num, tv_loss/style_num

def Gram(layer, batch_size):
    """ calcualte the gram matrix for the loss function """
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

if __name__ == '__main__':
  main()









