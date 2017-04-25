# -*- coding: utf-8 -*-
# author: K


import tensorflow as tf
import numpy as np
from os import walk
from os.path import join


class Generator:
	def __init__(self):
		self.reuse = False
	def build_graph(self, inputs, depth_size = 64, encoder_depth_times = [1, 2, 4, 8, 8, 8, 8, 8], decoder_depth_times = [8, 8, 8, 8, 8, 4, 2, 1]):
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
							[5, 5, encoder_input[i], encoder_output[i]],
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

						mean, variance = tf.nn.moments(conv, [0, 1, 2])

						bn = tf.nn.batch_normalization(conv, mean, variance, beta, None, 1e-5)

						outputs = tf.nn.relu(bn)


			with tf.variable_scope("decoder"):
				decoder_depths  = [depth_size * d for d in decoder_depth_times] + [3]
				length = len(decoder_depths)
				decoder_input = decoder_depths[:length]
				decoder_output = decoder_depths[1:]

				for i in range(length - 1):
					with tf.variable_scope('deconv%d' % i):
						w = tf.get_variable(
							'w',
							[5, 5, decoder_output[i], decoder_input[i]],
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

						mean, variance = tf.nn.moments(trans_conv, [0, 1, 2])

						bn = tf.nn.batch_normalization(trans_conv, mean, variance, beta, None, 1e-5)

						outputs = tf.nn.relu(bn)

		
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
						[5, 5, input_depths[i], output_depths[i]],
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
					mean, variance = tf.nn.moments(conv, [0, 1, 2])
					bn = tf.nn.batch_normalization(conv, mean, variance, beta, None, 1e-5)
			
					outputs = self.leaky_relu(bn)
		
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
	def __init__(self, batch_size, width = 256, lr = 0.0002):
		self.generator = Generator()
		self.discriminator = Discriminator()
		self.batch_size = batch_size
		self.lr = lr
		self.real_images_pair = tf.placeholder(tf.float32, shape = [self.batch_size, 2, width, width, 3])
	def build_graph(self):
		real_img_pair = self.real_images_pair
		real_img_p1, real_img_p2 = tf.split(real_img_pair, 2, axis = 1)

		real_img_p1 = tf.squeeze(real_img_p1, 1)
		real_img_p2 = tf.squeeze(real_img_p2, 1)


	def __call__(self):
		return self.build_graph()

class image_reader:
	def __init__():
		return None

	def read_img():
		return None


def read_data_pair():
	return None




def train(input_dir, target_dir, save_dir, batch_size = 32, lr = 5e-5, nb_epoch = 300):

	#imgs_pair = read_data_pair(input_dir, target_dir, target_size = [128, 128])

	return None




if __name__ == '__main__':
	train("./origin", "./target", "./save_dir")
