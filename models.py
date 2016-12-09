import tensorflow as tf
from tensorflow.contrib.layers import batch_norm
from ops import conv1x1, conv2x2, lrelu, relu, deconv2x2, sigmoid_xent
normal = tf.truncated_normal_initializer


class pix2pix(object):
    def __init__(self,
                 input_nc,
                 output_nc,
                 ngf,
                 ndf,
                 n_layers,
                 batch_sz,
                 lr,
                 beta1,
                 l1_lambda,
                 is_training):

        self.g_params = self.initialize_gen_params(input_nc, output_nc, ngf)
        self.d_params = self.initialize_disc_params(input_nc, output_nc, ndf, n_layers)

        self.keep_prob = tf.placeholder(tf.float32)

        self.real_data = tf.placeholder(tf.float32, [batch_sz, 256, 256, input_nc + output_nc])
        self.real_A = self.real_data[:, :, :, :input_nc]
        self.real_B = self.real_data[:, :, :, input_nc:]
        self.fake_B = self.defineG_unet(self.real_A, is_training)
        self.real_AB = tf.concat(3, [self.real_A, self.real_B])
        self.fake_AB = tf.concat(3, [self.real_A, self.fake_B])

        self.real_logits = self.defineD_n_layers(self.real_AB, n_layers, is_training)
        self.fake_logits = self.defineD_n_layers(self.fake_AB, n_layers, is_training)

        self.errD = tf.reduce_mean(sigmoid_xent(self.real_logits, tf.ones_like(self.real_logits)) + sigmoid_xent(self.fake_logits, tf.zeros_like(self.fake_logits)))

        self.errG = tf.reduce_mean(sigmoid_xent(self.fake_logits, tf.ones_like(self.fake_logits)))
        self.errL1 = tf.reduce_mean(tf.abs(self.fake_B - self.real_B))
        self.errG_total = self.errG + l1_lambda * self.errL1

        self.optim_D = tf.train.AdamOptimizer(lr, beta1=beta1)
        self.optim_G = tf.train.AdamOptimizer(lr, beta1=beta1)
        self.train_D = self.optim_D.minimize(self.errD, var_list=self.d_params)
        self.train_G = self.optim_G.minimize(self.errG_total, var_list=self.g_params)

    def initialize_gen_params(self, input_nc, output_nc, ngf):
        with tf.variable_scope('enc'):
            e1 = tf.get_variable('e1', [4, 4, input_nc, ngf], initializer=normal(stddev=0.02))
            e2 = tf.get_variable('e2', [4, 4, ngf, ngf * 2], initializer=normal(stddev=0.02))
            e3 = tf.get_variable('e3', [4, 4, ngf * 2, ngf * 4], initializer=normal(stddev=0.02))
            e4 = tf.get_variable('e4', [4, 4, ngf * 4, ngf * 8], initializer=normal(stddev=0.02))
            e5 = tf.get_variable('e5', [4, 4, ngf * 8, ngf * 8], initializer=normal(stddev=0.02))
            e6 = tf.get_variable('e6', [4, 4, ngf * 8, ngf * 8], initializer=normal(stddev=0.02))
            e7 = tf.get_variable('e7', [4, 4, ngf * 8, ngf * 8], initializer=normal(stddev=0.02))
            e8 = tf.get_variable('e8', [2, 2, ngf * 8, ngf * 8], initializer=normal(stddev=0.02))

        with tf.variable_scope('dec'):
            d1 = tf.get_variable('d1', [2, 2, ngf * 8, ngf * 8], initializer=normal(stddev=0.02))
            d2 = tf.get_variable('d2', [4, 4, ngf * 8, ngf * 8 * 2], initializer=normal(stddev=0.02))
            d3 = tf.get_variable('d3', [4, 4, ngf * 8, ngf * 8 * 2], initializer=normal(stddev=0.02))
            d4 = tf.get_variable('d4', [4, 4, ngf * 8, ngf * 8 * 2], initializer=normal(stddev=0.02))
            d5 = tf.get_variable('d5', [4, 4, ngf * 4, ngf * 8 * 2], initializer=normal(stddev=0.02))
            d6 = tf.get_variable('d6', [4, 4, ngf * 2, ngf * 4 * 2], initializer=normal(stddev=0.02))
            d7 = tf.get_variable('d7', [4, 4, ngf, ngf * 2 * 2], initializer=normal(stddev=0.02))
            d8 = tf.get_variable('d8', [4, 4, output_nc, ngf * 2], initializer=normal(stddev=0.02))

            g_params = [e1, e2, e3, e4, e5, e6, e7, e8, d1, d2, d3, d4, d5, d6, d7, d8]

        return g_params

    def initialize_disc_params(self, input_nc, output_nc, ndf, n_layers):
        with tf.variable_scope('disc'):
            d0 = tf.get_variable('d0', [4, 4, input_nc + output_nc, ndf], initializer=normal(stddev=0.02))
            d_params = [d0]

            nf_mult = 1
            for n in range(1, n_layers + 1):
                nf_mult_prev = nf_mult
                nf_mult = min(pow(2, n), 8)
                d_params.append(tf.get_variable('d%d' % (n), [4, 4, ndf * nf_mult_prev, ndf * nf_mult], initializer=normal(stddev=0.02)))

            d_params.append(tf.get_variable('d%d' % (n + 1), [4, 4, ndf * nf_mult, 1], initializer=normal(stddev=0.02)))

        return d_params

    def defineG_unet(self, inp, is_training=True):
        e1, e2, e3, e4, e5, e6, e7, e8, d1, d2, d3, d4, d5, d6, d7, d8 = self.g_params

        enc1 = conv2x2(inp, e1)
        enc2 = batch_norm(conv2x2(lrelu(enc1), e2), is_training=is_training)
        enc3 = batch_norm(conv2x2(lrelu(enc2), e3), is_training=is_training)
        enc4 = batch_norm(conv2x2(lrelu(enc3), e4), is_training=is_training)
        enc5 = batch_norm(conv2x2(lrelu(enc4), e5), is_training=is_training)
        enc6 = batch_norm(conv2x2(lrelu(enc5), e6), is_training=is_training)
        enc7 = batch_norm(conv2x2(lrelu(enc6), e7), is_training=is_training)
        enc8 = batch_norm(conv2x2(lrelu(enc7), e8), is_training=is_training)

        dec1 = batch_norm(deconv2x2(relu(enc8), d1), is_training=is_training)
        dec1 = tf.nn.dropout(dec1, self.keep_prob)
        dec1 = tf.concat(3, [dec1, enc7])
        dec2 = batch_norm(deconv2x2(relu(dec1), d2), is_training=is_training)
        dec2 = tf.nn.dropout(dec2, self.keep_prob)
        dec2 = tf.concat(3, [dec2, enc6])
        dec3 = batch_norm(deconv2x2(relu(dec2), d3), is_training=is_training)
        dec3 = tf.nn.dropout(dec3, self.keep_prob)
        dec3 = tf.concat(3, [dec3, enc5])
        dec4 = batch_norm(deconv2x2(relu(dec3), d4), is_training=is_training)
        dec4 = tf.concat(3, [dec4, enc4])
        dec5 = batch_norm(deconv2x2(relu(dec4), d5), is_training=is_training)
        dec5 = tf.concat(3, [dec5, enc3])
        dec6 = batch_norm(deconv2x2(relu(dec5), d6), is_training=is_training)
        dec6 = tf.concat(3, [dec6, enc2])
        dec7 = batch_norm(deconv2x2(relu(dec6), d7), is_training=is_training)
        dec7 = tf.concat(3, [dec7, enc1])
        dec8 = tf.tanh(deconv2x2(relu(dec7), d8))

        return dec8

    def defineD_n_layers(self, inp, n_layers, is_training=True):
        disc = lrelu(conv2x2(inp, self.d_params[0]))

        for n in range(1, n_layers):
            disc = lrelu(batch_norm(conv2x2(disc, self.d_params[n]), is_training=is_training))

        disc = lrelu(batch_norm(conv1x1(disc, self.d_params[n_layers]), is_training=is_training))
        out = conv1x1(disc, self.d_params[n_layers + 1])

        return out
