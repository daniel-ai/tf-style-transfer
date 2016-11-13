"""
tensorflow implementation of the paper
A Learned Representation For Artistic Style
"""
import tensorflow as tf
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime
import os, sys, scipy.io
import loss, nn_build

# description of neural network layers in the form of
# 'layer_type-input_channels-output_channels-kernel_height-kernel_width
# -stride_height-stride-width'
# if the parameters are needed for the corresponding layer_type
# layer_type norm = instance normalization
# layer_type res = a residual block of [conv, norm, relu, conv, norm]
# layer_type tanh_resc = (tanh+1)*255/2
MODELDESC = (
    'conv2d-9-9-3-32-1-1, s_norm-32, relu, conv2d-3-3-32-64-2-2, s_norm-64, '
    'relu, conv2d-3-3-64-128-2-2, s_norm-128, relu, res-3-3-128-128-1-1, '
    'res-3-3-128-128-1-1, res-3-3-128-128-1-1, res-3-3-128-128-1-1, '
    'res-3-3-128-128-1-1, tconv2d-3-3-128-64-2-2, s_norm-64, relu, '
    'tconv2d-3-3-64-32-2-2, s_norm-32, relu, tconv2d-9-9-32-3-1-1, tanh_resc')
STYLES_TRAINED = ['1','2','3','4','5','6']


def args():
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, default='train',
                        help='name of the style transfer model')
    parser.add_argument('--content_dir', type=str, default='./image_input/',
                        help='content images (c_*.jpg) to be stylized')
    parser.add_argument('--style_names', nargs='+', type=str,
                        default=STYLES_TRAINED,
                        help='styles names to be applied in stylization')
    parser.add_argument('--style_dir', type=str, default='./image_input/',
                        help='styple images (s_*.jpg) for training')
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
    """ train models or stylize images """
    options = args()
    with tf.Graph().as_default():
        if options.stylize:
            stylize(options.model_name, options.content_dir,
                    options.style_names, options.model, options.output_dir)
        else:
            train(options.model_name, options.style_dir, options.train_dir,
                  options.image_size, options.global_step, options.lr,
                  options.batch_size, options.val_size, options.vggfile,
                  options.model, options.max_iteration,
                  options.eval_iteration, options.save_iteration)


def train(model_name, style_dir, train_dir, train_image_size, global_step,
          lr, batch_size, val_size, vggfile, model, max_iteration,
          eval_iteration, save_iteration):
    """ train a single style tranfer neural network for a group of styles

    see args help for definition

    """
    style_images = {}
    style_names = []
    for root, dirs, files in os.walk(style_dir):
        for filenm in files:
            if filenm[0:2] == 's_':
                style_image = scipy.misc.imread(os.path.join(root,filenm))
                style_names.append(filenm[2:-4])
                style_images[style_names[-1]] = tf.to_float(
                    style_image[np.newaxis,:,:,:])
    print 'Styles '+', '.join(style_names)+' used for training.'

    # create a tf placeholder for training content images
    input_shape = [batch_size] + train_image_size + [3]
    content_image = tf.placeholder(
        tf.float32, input_shape, name='content_image')
    style_net = nn_build.Network(
        MODELDESC, model_name, style_names, trainable=True)
    output_images = style_net.images(content_image, style_names)

    c_loss, s_loss, tv_loss = loss.loss_vgg(
        style_images, content_image, output_images, vggfile)
    total_loss = c_loss + s_loss + tv_loss
    # use Adam optimizer with gradient clipping
    optimizer = tf.train.AdamOptimizer(lr)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 1)
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
                print 'Session file saved to {}-{}.'.format(model, global_step)
                sys.stdout.flush()
        print 'Training finished (global step = {}).'.format(global_step)


def stylize(model_name, content_dir, style_names, model, output_dir):
    """ stylize image using pre-trained nerual network

    see args help for definition

    """
    content_images = {}
    for root, dirs, files in os.walk(content_dir):
        for filenm in files:
            if filenm[0:2] == 'c_':
                content_image = scipy.misc.imread(os.path.join(root, filenm))
                c_name = filenm[2:-4]
                content_images[c_name] = tf.to_float(content_image[np.newaxis,:,:,:])

    style_net = nn_build.Network(
        MODELDESC, model_name, style_names, trainable=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model)
        print 'Stylizing {} images'.format(len(content_images))
        for c_name in content_images:
            output_images = style_net.images(
                content_images[c_name], style_names)
            for style in style_names:
                output_image = Image.fromarray(
                    output_images[style].eval()[0].astype(np.uint8), "RGB")
                output_image.save(
                    output_dir+'stylized_{}_{}.jpg'.format(c_name, style))
    print 'Stylized {} images into {} styles.'.format(
        len(content_images), len(style_names))


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


if __name__ == '__main__':
  main()









