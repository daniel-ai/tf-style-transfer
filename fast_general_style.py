import tensorflow as tf
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from datetime import datetime
import os, sys, scipy.io
import loss, nn_build

# description of model layers
MODELDESC = 'conv2d-9-9-3-32-1-1, s_norm-32, relu, conv2d-3-3-32-64-2-2, s_norm-64, relu, conv2d-3-3-64-128-2-2, s_norm-128, relu, res-3-3-128-128-1-1, res-3-3-128-128-1-1, res-3-3-128-128-1-1, res-3-3-128-128-1-1, res-3-3-128-128-1-1, tconv2d-3-3-128-64-2-2, s_norm-64, relu, tconv2d-3-3-64-32-2-2, s_norm-32, relu, tconv2d-9-9-32-3-1-1, tanh_res'
STYLES_TRAINED = ['1','2','3','4','5','6']

def args():
    parser = ArgumentParser()
    parser.add_argument('--content_dir', help='location of the content images (in the form of c_*.jpg) to be stylized', type=str, default='./image_input/')
    parser.add_argument('--style_names', help='names of the styles to be applied to the content images', nargs='+', type=str, default=STYLES_TRAINED)
    parser.add_argument('--style_dir', help='location of the training styple images (in the form of s_*.jpg)', type=str, default='./image_input/')
    parser.add_argument('--train_dir', help='location of the training content images', type=str, default='./COCO/')
    parser.add_argument('--img_size', help='[h, w] of training images', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--vggfile', help='location of pre-trained vgg model', type=str, default='./models/imagenet-vgg-verydeep-16.mat')
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--batch_size', help='training batch size', type=int, default=1)
    parser.add_argument('--val_size', help='number of batches used for validation', type=int, default=10)
    parser.add_argument('--global_step', help='global count of training steps', type=int, default=0)
    parser.add_argument('--max_iteration', help='maximum number of iterations for training ', type=int, default=80000)
    parser.add_argument('--eval_iteration', help='evalute loss per number of iterations', type=int, default=1000)
    parser.add_argument('--save_iteration', help='save session file per number of iterations', type=int, default=5000)
    parser.add_argument('--model', help='path to save/restore model', type=str, default='./models/model_general')
    parser.add_argument('--model_name', help='name of the model used in scope', type=str, default='train')
    parser.add_argument('--stylize', help='indicator to stylize images', action='store_true')
    parser.add_argument('--output', help='path to save stylized images', type=str, default='./image_output/')
    options = parser.parse_args()
    return options

def main():
    options = args()
    with tf.Graph().as_default():
        if options.stylize:
            stylize(options.model_name, options.content_dir, options.style_names, options.model, options.output)
        else:
            train(options.model_name, options.style_dir, options.train_dir, options.img_size, options.global_step, options.lr, options.batch_size, options.val_size, options.vggfile, options.model, options.max_iteration, options.eval_iteration, options.save_iteration)

def train(model_name, style_dir, train_dir, train_img_size, global_step, lr, batch_size, val_size, vggfile, model, max_iteration, eval_iteration, save_iteration):
    """ train a style-tranfer neural network given a specific style

    """
    s_imgs = {}
    style_names = []
    for root, dirs, files in os.walk(style_dir):
        for filenm in files:
            if filenm[0:2] == 's_':
                s_img = scipy.misc.imread(os.path.join(root,filenm))
                style_names.append(filenm[2:-4])
                s_imgs[style_names[-1]] = tf.to_float(s_img[np.newaxis,:,:,:])
    print 'Styles '+', '.join(style_names)+' used for training.'

    input_shape = [batch_size] + train_img_size + [3]
    c_img = tf.placeholder(tf.float32, input_shape, name='content_image')
    style_net = nn_build.Network(MODELDESC, model_name, style_names, trainable=True)
    o_imgs = style_net.images(c_img, style_names)

    c_loss, s_loss, tv_loss = loss.loss_vgg(s_imgs, c_img, o_imgs, vggfile)
    total_loss = c_loss + s_loss + tv_loss
    optimizer = tf.train.AdamOptimizer(lr)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 1)
    train_op = optimizer.apply_gradients(zip(grads, tvars))

    train_img = train_set(train_dir)
    # first val_size of batches are used as the validation set
    val_set = [train_img(batch_size) for k in range(val_size)]
    saver = tf.train.Saver()
    with tf.Session() as sess:
        if global_step == 0:
            sess.run(tf.initialize_all_variables())
        else:
            saver.restore(sess, model+'-{}'.format(global_step))
        train_img.index = global_step % (train_img.number - batch_size*val_size) + batch_size*val_size
        print 'Started training with {} trainable tensors:'.format(len(tf.trainable_variables()))
        sys.stdout.flush()
        time_start = datetime.now()
        for i in range(max_iteration):
            if train_img.index+batch_size > train_img.number:
                train_img.index = batch_size*val_size
            img = train_img(batch_size)
            sess.run(train_op, {c_img: img})
            global_step += batch_size
            if  (i+1) % eval_iteration == 0 or (i+1) == max_iteration:
                time_end = datetime.now()
                iterations = eval_iteration if (i+1) % eval_iteration == 0 else (i+1) % eval_iteration
                val_losses = np.stack([sess.run([c_loss, s_loss, tv_loss], {c_img: val_img}) for val_img in val_set])
                [val_c_loss, val_s_loss, val_tv_loss] = np.mean(val_losses, axis=0)/batch_size
                print 'iteration %d: c_loss = %.1f, s_loss = %.1f, tv_loss = %.1f, ave_time = %s' %(i+1, val_c_loss, val_s_loss, val_tv_loss, (time_end-time_start)/iterations)
                sys.stdout.flush()
                time_start = datetime.now()
            if  (i+1) % save_iteration == 0 or (i+1) == max_iteration:
                saver.save(sess, model, global_step, write_meta_graph=False)
                print 'Session file saved to {}-{}.'.format(model, global_step)
                sys.stdout.flush()
        print 'Training finished (global step = {}).'.format(global_step)

def stylize(model_name, content_dir, style_names, model, output):
    """ stylize an image using pre-trained nerual network

    """
    c_imgs = {}
    for root, dirs, files in os.walk(content_dir):
        for filenm in files:
            if filenm[0:2] == 'c_':
                c_img = scipy.misc.imread(os.path.join(root, filenm))
                name = filenm[2:-4]
                c_imgs[name] = tf.to_float(c_img[np.newaxis,:,:,:])
    print 'Stylizing {} content images'.format(len(c_imgs))
    style_net = nn_build.Network(MODELDESC, model_name, style_names, trainable=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model)
        for name in c_imgs.keys():
            o_imgs = style_net.images(c_imgs[name], style_names)
            for style in style_names:
                o_img = Image.fromarray(o_imgs[style].eval()[0].astype(np.uint8), "RGB")
                o_img.save(output+'stylized_{}_{}.jpg'.format(name, style))
    print '{} stylized image saved to {}.'.format(len(c_imgs)*len(style_names), output)

class train_set:
    """ a training data feeder

    """
    def __init__(self, train_dir):
        self.files = [os.path.join(root,filenm) for root, dirs, files in os.walk(train_dir) for filenm in files if filenm[-4:] == '.jpg']
        np.random.shuffle(self.files)
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









