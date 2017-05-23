# -*- coding: utf-8 -*-
# author: K


import tensorflow as tf
import numpy as np
from os import walk
from os.path import join
from PIL import Image

class Generator:
	def __init__(self):
		self.reuse = False
	def build_graph(self, inputs, depth_size = 64, encoder_depth_times = [1, 2, 4, 8, 8, 8], decoder_depth_times = [8, 8, 8, 4, 2, 1]):
		# The Generator is a Encoder-Decoder Model
		outputs = inputs

		with tf.variable_scope("generator", reuse = self.reuse):
			with tf.variable_scope("encoder"):
				encoder_depths  = [3] + [depth_size * d for d in encoder_depth_times]
				length = len(encoder_depths)
				encoder_input = encoder_depths[:length]
				encoder_output = encoder_depths[1:]

				for i in range(length - 1):
					with tf.variable_scope('conv%d' % i):

						w = tf.get_variable(
							'w',
							[3, 3, encoder_input[i], encoder_output[i]],
							tf.float32,
							tf.truncated_normal_initializer(stddev = 0.02)
						)

						beta = tf.get_variable(
							'beta',
							[encoder_output[i]],
							tf.float32,
							tf.zeros_initializer()
						)


						conv = tf.nn.conv2d(outputs, w, [1, 2, 2, 1], 'SAME')

						#mean, variance = tf.nn.moments(conv, [0, 1, 2])

						#bn = tf.nn.batch_normalization(conv, mean, variance, beta, None, 1e-5)

						outputs = tf.nn.relu(conv)

			with tf.variable_scope("decoder"):
				decoder_depths  = [depth_size * d for d in decoder_depth_times] + [3]
				length = len(decoder_depths)
				decoder_input = decoder_depths[:length]
				decoder_output = decoder_depths[1:]

				for i in range(length - 1):
					with tf.variable_scope('deconv%d' % i):
						w = tf.get_variable(
							'w',
							[3, 3, decoder_output[i], decoder_input[i]],
							tf.float32,
							tf.truncated_normal_initializer(stddev = 0.02)
						)

						beta = tf.get_variable(
								'beta',
								[decoder_output[i]],
								tf.float32,
								tf.zeros_initializer()
						)

						output_shape = outputs.get_shape().as_list()[1:3]

						trans_conv = tf.nn.conv2d_transpose(
							outputs, 
							w, 
							[int(outputs.get_shape()[0]), output_shape[0] * 2, output_shape[1] * 2, decoder_output[i]],
							[1, 2, 2, 1]
						)

						#mean, variance = tf.nn.moments(trans_conv, [0, 1, 2])

						#bn = tf.nn.batch_normalization(trans_conv, mean, variance, beta, None, 1e-5)

						outputs = tf.nn.relu(trans_conv)

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')
		
		return outputs

	def __call__(self, inputs):
		return self.build_graph(inputs)



class Discriminator:
	def __init__(self):
		self.reuse = False
	def build_graph(self, inputs, depths = [64, 128, 256, 512]):

		outputs = inputs

		with tf.variable_scope("discriminator", reuse = self.reuse):

			depths = [6] + depths
			length = len(depths)
			input_depths = depths[:length]
			output_depths = depths[1:]

			for i in range(length - 1):
				with tf.variable_scope("conv%d" % i):
					w = tf.get_variable(
						'w',
						[3, 3, input_depths[i], output_depths[i]],
						tf.float32,
						tf.truncated_normal_initializer(stddev = 0.02)
					)

					beta = tf.get_variable(
						'beta',
						[output_depths[i]],
						tf.float32,
						tf.truncated_normal_initializer(stddev = 0.02)
					)
					
					conv = tf.nn.conv2d(outputs, w, [1, 2, 2, 1], 'SAME')
					#mean, variance = tf.nn.moments(conv, [0, 1, 2])
					#bn = tf.nn.batch_normalization(conv, mean, variance, beta, None, 1e-5)
			
					outputs = self.leaky_relu(conv)
		
			with tf.variable_scope("classify"):
				tensor_shape = outputs.get_shape().as_list()

                                flatten_dim = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]

                                w = tf.get_variable(
                                        'w',
                                        [flatten_dim, 1],
                                        tf.float32,
                                        tf.truncated_normal_initializer(stddev = 0.02)
                                )

                                b = tf.get_variable(
                                        'b',
                                        [1],
                                        tf.float32,
                                        tf.zeros_initializer()
                                )

                                outputs = tf.reshape(outputs, [-1, flatten_dim])

                                outputs = tf.nn.bias_add(tf.matmul(outputs, w), b)

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'discriminator')
		return outputs

	def leaky_relu(self, x, alpha = 0.2):
		return tf.maximum(x, x * alpha)

	def __call__(self, inputs):
		return self.build_graph(inputs)



class cLSGAN:
	def __init__(self, batch_size, image_size = 64, lr = 5e-5):
		self.generator = Generator()
		self.image_size = image_size
		self.discriminator = Discriminator()
		self.batch_size = batch_size
		self.lr = lr
		self.real_images_pair = tf.placeholder(tf.float32, shape = [self.batch_size, 2, image_size, image_size, 3])
	def build_graph(self, is_peason_div = False):
		'''
		def build_loss(d_logits_fake, d_logits_real, a, b, c):
			d_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_real - b)) + tf.reduce_mean(tf.square(d_logits_fake - a))
			
			lsgan_loss = 0.5 * tf.reduce_mean(tf.square(d_logits_fake - c))

			return d_loss, lsgan_loss

		real_img_pair = self.real_images_pair
		real_img_p1, real_img_p2 = tf.split(real_img_pair, 2, axis = 1)

		real_img_source = tf.squeeze(real_img_p1, 1)
		real_img_target = tf.squeeze(real_img_p2, 1)

		g_logits = self.generator(real_img_source)

		fake_pair = tf.concat([real_img_source, g_logits], axis = 3)

		real_pair = tf.concat([real_img_source, real_img_target], axis = 3)

		d_logits_fake = self.discriminator(fake_pair)

		d_logits_real = self.discriminator(real_pair)

		if is_peason_div:
			d_loss, lsgan_loss = build_loss(d_logits_fake, d_logits_real, -1, 1, 0)
		else:
			d_loss, lsgan_loss = build_loss(d_logits_fake, d_logits_real, 0, 1, 1)

		d_optim = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(d_loss, var_list = self.discriminator.variables)

		lsgan_optim = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(lsgan_loss, var_list = self.generator.variables)

		'''
		real_img_pair = self.real_images_pair
		real_img_p1, real_img_p2 = tf.split(real_img_pair, 2, axis = 1)

		real_img_source = tf.squeeze(real_img_p1, 1)
		real_img_target = tf.squeeze(real_img_p2, 1)
	
		g_logits = self.generator(real_img_source)

		fake_pair = tf.concat([real_img_source, g_logits], axis = 3)
		real_pair = tf.concat([real_img_source, real_img_target], axis = 3)

		d_logits_fake = self.discriminator(fake_pair)

		d_logits_real = self.discriminator(real_pair)

		d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.zeros([self.batch_size, 1])))

		d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_real, labels = tf.ones([self.batch_size, 1])))

		d_loss = d_loss_fake + d_loss_real

		lsgan_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = d_logits_fake, labels = tf.ones([self.batch_size, 1])))

		d_optim = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(d_loss, var_list = self.discriminator.variables)

		lsgan_optim = tf.train.AdamOptimizer(learning_rate = self.lr).minimize(lsgan_loss, var_list = self.generator.variables)

		return d_optim, lsgan_optim, d_loss, lsgan_loss, g_logits


	def sample_image(self, real_img_pair, sample_size):

		real_img_p1, real_img_p2 = tf.split(real_img_pair, 2, axis = 1)

		real_img_source = tf.squeeze(real_img_p1, 1)
		real_img_target = tf.squeeze(real_img_p2, 1)

		gen_images = self.generator(real_img_source)

		ims = tf.multiply(gen_images, 255.0)

		return ims


def load_img(path, grayscale = False, target_size = None):
        img = Image.open(path)

        if grayscale:
                if img.mode != 'L':
                        img = img.convert('L')
        else:
                if img.mode != 'RGB':
                        img = img.convert('RGB')

        if target_size:
                wh_tuple = (target_size[1], target_size[0])
                if img.size != wh_tuple:
                        img = img.resize(wh_tuple)
        return img

def img_to_array(img):
        x = np.asarray(img, dtype = np.float32)
        return x

def array_to_img(x, data_format=None, scale=True):
	x = np.asarray(x, dtype = np.float32)
	if x.ndim != 3:
		raise ValueError('Expected image array to have rank 3 (single image). '
			 'Got array with shape:', x.shape)

	if data_format == 'channels_first':
		x = x.transpose(1, 2, 0)
	if scale:
		x = x + max(-np.min(x), 0)
		x_max = np.max(x)
		if x_max != 0:
	    		x /= x_max
		x *= 255
	if x.shape[2] == 3:
		# RGB
		return Image.fromarray(x.astype('uint8'), 'RGB')
	elif x.shape[2] == 1:
		# grayscale
		return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
	else:
		raise ValueError('Unsupported channel number: ', x.shape[2])

def read_data_pair(source_dir, target_dir, target_size):
	imgs_pair = []

	for root, sub, files in walk(source_dir):
		for f in files:
			source_img = img_to_array(load_img(join(source_dir, f), grayscale = False, target_size = target_size))
			target_img = img_to_array(load_img(join(target_dir, f), grayscale = False, target_size = target_size))

			source_img = source_img.reshape((1,) + source_img.shape)

			target_img = target_img.reshape((1,) + target_img.shape)

			imgs_pair.append(np.concatenate((source_img, target_img), axis = 0))
	
	imgs_pair = np.asarray(imgs_pair) / 255.0

	return imgs_pair


def train(input_dir, target_dir, save_dir, batch_size = 1, lr = 5e-5, sample_size nb_epoch = 1000):
	print "[*] Start our magic"

	print "[*] Read the paired data"

	imgs_pair = read_data_pair(input_dir, target_dir, target_size = [64, 64])

	print "[*] Data readed"

	nb_samples = imgs_pair.shape[0]

	#nb_batches = int(nb_samples / batch_size)

	nb_batches = 32 / batch_size
	
	cLSGAN_model = cLSGAN(batch_size = batch_size, lr = lr)

	d_optim, lsgan_optim, d_loss, lsgan_loss, g_logits = cLSGAN_model.build_graph(False)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		for epoch in range(1, nb_epoch + 1):
			print "Training Epoch %d" % epoch

			for i in range(nb_batches):

				for j in range(5):
			
					real_images_pair = imgs_pair[np.random.randint(nb_samples, size = batch_size)]

					sess.run(d_optim, feed_dict = {cLSGAN_model.real_images_pair: real_images_pair})

				sess.run(lsgan_optim, feed_dict = {cLSGAN_model.real_images_pair: real_images_pair})

			if epoch % 5 == 0:

				sampled_imgs = sess.run(cLSGAN_model.sample_image(real_images_pair))
	
				#res = sess.run(g_logits, feed_dict = {cLSGAN_model.real_images_pair: real_images_pair})

				idx = 0
		
				for each_im in sampled_imgs:
					img = sess.run(tf.image.encode_jpeg(each_im))

					with open(save_dir + '/sample%d_%d.jpeg' % (epoch, idx), "wb") as f:
						f.write(img)

					idx += 1


if __name__ == '__main__':
	train("./source_dir", "./target_dir", "./save_dir")
