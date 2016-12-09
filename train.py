from datetime import datetime
from models import pix2pix
from data_loader import data_provider, load_random_samples, save_images
import tensorflow as tf
import argparse
import sys
import numpy as np
from rng import *


def train(args, sess):
    dataset_name = args.dataset_name
    batch_size = args.batch_size
    load_size = args.load_size
    fine_size = args.fine_size
    flip = args.flip
    shuffle = True
    max_epoch = args.max_epoch
    save_epoch_freq = args.save_epoch_freq
    disp_freq = args.disp_freq
    print_freq = args.print_freq
    checkpoints_dir = args.checkpoints_dir
    sample_dir = args.sample_dir

    ngf = args.ngf
    ndf = args.ndf
    input_nc = args.input_nc
    output_nc = args.output_nc
    n_layers_D = args.n_layers_D

    lr = args.lr
    beta1 = args.beta1
    l1_lambda = args.l1_lambda

    model = pix2pix(input_nc, output_nc, ngf, ndf, n_layers_D, batch_size, lr, beta1, l1_lambda, is_training=True)
    sess.run(tf.initialize_all_variables())
    saver = tf.train.Saver()

    iter_counter = 0

    d_err_list = []
    g_err_list = []
    l1_err_list = []
    for k in range(max_epoch):
        for real_data in data_provider(dataset_name, batch_size, load_size, fine_size, flip, shuffle):
            real_logits, fake_logits = sess.run([model.real_logits, model.fake_logits], feed_dict={model.real_data: real_data, model.keep_prob: 0.5})

            d_err, _ = sess.run([model.errD, model.train_D], feed_dict={model.real_data: real_data, model.keep_prob: 0.5})
            g_err, l1_err, _ = sess.run([model.errG, model.errL1, model.train_G], feed_dict={model.real_data: real_data, model.keep_prob: 0.5})

            d_err_list.append(d_err)
            g_err_list.append(g_err)
            l1_err_list.append(l1_err)

            iter_counter += 1

            if iter_counter % print_freq == 0:
                time_now = datetime.now()
                time_tag = '[%02d.%02d.%02d]' % (time_now.hour, time_now.minute, time_now.second
                print time_tag + " epoch %3d, iter %d, errD: %.7f, errG: %.7f, errL1: %.7f" % (k, iter_counter, np.mean(d_err_list[-print_freq:]), np.mean(g_err_list[-print_freq:]), np.mean(l1_err_list[-print_freq:]))

            if iter_counter % disp_freq == 0:
                print "    Sampling..."
                sample_images = load_random_samples(dataset_name, batch_size, fine_size)
                samples, d_loss, g_loss, l1_loss = sess.run([model.fake_B, model.errD, model.errG, model.errL1], feed_dict={model.real_data: sample_images, model.keep_prob: 1.})
                save_images(sample_images, samples, sample_dir, iter_counter)
                print "    samples, errD: %.7f, errG: %.7f, errL1: %.7f" % (d_loss, g_loss, l1_loss)

        if (k + 1) % save_epoch_freq == 0:
            print "    Saving..."
            saver.save(sess, checkpoints_dir + '/pix2pix', global_step=k)

    return model


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='facades')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--load_size', default=286, type=int, help='scale images to this size')
    parser.add_argument('--fine_size', default=256, type=int, help='then crop to this size')
    parser.add_argument('--ngf', default=64, type=int, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', default=64, type=int, help='# of discrim filters in first conv layer')
    parser.add_argument('--input_nc', default=3, type=int, help='# of input image channels')
    parser.add_argument('--output_nc', default=3, type=int, help='# of output image channels')
    parser.add_argument('--max_epoch', default=200, type=int)
    parser.add_argument('--lr', default=0.0002)
    parser.add_argument('--beta1', default=0.5, help='momentum term of adam')
    parser.add_argument('--flip', default=1, help='if flip the images for data argumentation')
    parser.add_argument('--phase', default='train')
    parser.add_argument('--preprocess', default='regular')
    parser.add_argument('--save_epoch_freq', default=50)
    parser.add_argument('--disp_freq', default=500)
    parser.add_argument('--print_freq', default=50)
    parser.add_argument('--checkpoints_dir', default='./checkpoints')
    parser.add_argument('--sample_dir', default='./samples')
    parser.add_argument('--which_model_netD', default='basic')
    parser.add_argument('--which_model_netG', default='unet')
    parser.add_argument('--n_layers_D', default=3, help="only used if which_model_D=='n_layers'")
    parser.add_argument('--l1_lambda', default=100, help='weight on L1 term in objective')

    args = parser.parse_args(argv[1:])
    sess = tf.Session()
    model = train(args, sess)


if __name__ == "__main__":
    main(sys.argv)
