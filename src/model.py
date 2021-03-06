import tensorflow as tf
import numpy as np
import matplotlib, time, os, random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.misc import imread



class DCGAN(object):
    """docstring for DCGAN"""
    def __init__(self, sess, batch_size=128, latent_dim=100, learning_rate=5e-5, steps=6000, drop_rate=0.5,
    	model_path='../model/', vis_path='../result/', data_path='../anime-faces', load_checkpoint=None,
    	tensorboard_path='../TensorBoard'):

        self.sess = sess
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.base_learning_rate = learning_rate
        self.steps = steps + 1
        self.momentum = 0.9
        self.dropout = 1 - drop_rate
        self.model_path = model_path
        self.vis_path = vis_path
        self.data_path = data_path
        self.tensorboard_path = tensorboard_path
        self.test_vec = np.random.normal(-1, 1, [25, self.latent_dim]).astype(np.float32) # used for visualization during training
        self.load_checkpoint = load_checkpoint
        self.start = 1

        if not os.path.isdir(self.model_path):
        	os.mkdir(self.model_path)
        if not os.path.isdir(self.vis_path):
        	os.mkdir(self.vis_path)

        self.ReadData()
        self.CreateModel()

        if not os.path.isdir(self.model_path):
        	os.mkdir(self.model_path)

        if not os.path.isdir(self.tensorboard_path):
        	os.mkdir(self.tensorboard_path)

    def random_noise(self):

    	return np.random.normal(-1, 1, [self.batch_size, self.latent_dim]).astype(np.float32)

    def noisy_label(self, ones):

    	if ones:
    		return np.random.uniform(0.9, 1, [self.batch_size, 1]).astype(np.float32)

    	else:

    		return np.random.uniform(0, 0.1, [self.batch_size, 1]).astype(np.float32)

    def deconv2d(self, inputs, filters, kernel_size, alpha=0.2, padding='same', sigmoid=False, strides=2):

        def lrelu(x):
            return tf.nn.leaky_relu(x, alpha=alpha)

        if not sigmoid:
            activation = lrelu
        else:
            # activation = tf.nn.sigmoid
            activation = tf.nn.tanh

        tmp = tf.layers.conv2d_transpose(
                inputs=inputs,
                filters=filters,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.02),
                kernel_size=kernel_size,
                strides=strides,
                activation=activation,
                padding=padding
                )

        return tmp

    def conv2d(self, inputs, filters, kernel_size, alpha=0.2, padding='same', strides=2):
        tmp = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.02),
                kernel_size=kernel_size,
                padding=padding,
                strides=strides)
        tmp = tf.nn.leaky_relu(tmp, alpha=alpha)
        return tmp

    def generator(self, z, keep_prob, is_training):
        with tf.variable_scope('generator') as scope:
            print ('=======================================================================')
            print ('GENERATOR\n     {:50} {}'.format('Layer', 'Shape'))
            # shape 100
            x = z
            print ('{:50} {}'.format(x.name, x.shape))
            # project and reshape 1*1*1024
            x = tf.layers.dense(x, units=1*1*1024, activation=tf.nn.leaky_relu, 
            	kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.02))
            x = tf.layers.dropout(x, keep_prob)
            x = tf.reshape(x, shape=[-1, 1, 1, 1024])
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 3*3*1024
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,  512, [3, 3], padding='valid')
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 6*6*512
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,  256, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 12*12*256
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,  128, [5, 5])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 24*24*128
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,   64, [5, 5])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 48*48*64
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,   32, [5, 5])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 96*96*3
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,    3, [7, 7], sigmoid=True)
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}\n'.format(x.name, x.shape))

        return x

    def discriminator(self, image, keep_prob, is_training, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            print ('=======================================================================')
            print ('DISCRIMINATOR\n     {:50} {}'.format('Layer', 'Shape'))
            # shape 96*96*3
            x = image
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 32*32*64
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,   32, [5, 5], strides=3)
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 16*16*128
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,   64, [5, 5])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 8*8*256
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,  128, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 4*4*512
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,  256, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 2*2*1024
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,  512, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 1*1*1
            # x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            # x = self.conv2d(x,  512, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            # print ('{:50} {}'.format(x.name, x.shape))
            # reshape
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, units=1)
            print ('{:50} {}\n'.format(x.name, x.shape))

            logits = x

        return logits, tf.nn.sigmoid(logits)

    def CreateModel(self):

    	# placeholders
        self.image = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        self.noise = tf.placeholder(tf.float32, [None, self.latent_dim], name='noise')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # models
        self.G                          = self.generator(self.noise, self.dropout, self.is_training)
        self.D_real_logits, self.D_real = self.discriminator(self.image, self.dropout, self.is_training)
        self.D_fake_logits, self.D_fake = self.discriminator(self.G, self.dropout, self.is_training, reuse=True)

        self.sum_img = tf.summary.image('generated', self.G, max_outputs=4)
        self.sum_real_out = tf.summary.histogram('real_img_result_the_closer_to_0_the_better', self.D_real)
        self.sum_fake_out = tf.summary.histogram('fake_img_result_the_closer_to_1_the_better', self.D_fake)
        self.sum_gen_out  = tf.summary.histogram('gen__img_result_the_closer_to_0_the_better', self.D_fake)

        # define loss
        def sigmoid_cross_entropy_with_logits(y, x):
        	return tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=x)

        with tf.name_scope('Loss'):
	        # self.G_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(tf.ones_like(self.D_fake), self.D_fake_logits))
	        # self.D_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(tf.ones_like(self.D_real), self.D_real_logits))
	        # self.D_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(tf.zeros_like(self.D_fake), self.D_fake_logits))
	        
	        # use noisy labels for discriminator only
	        # self.G_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(tf.ones_like(self.D_fake), self.D_fake_logits))
	        # self.D_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.noisy_label(ones=True), self.D_real_logits))
	        # self.D_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.noisy_label(ones=False), self.D_fake_logits))
	        
	        # flip lables
	        self.G_loss = tf.reduce_mean(sigmoid_cross_entropy_with_logits(tf.zeros_like(self.D_fake), self.D_fake_logits))
	        self.D_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.noisy_label(ones=False), self.D_real_logits))
	        self.D_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.noisy_label(ones=True), self.D_fake_logits))
	        
	        self.D_loss = self.D_loss_real + self.D_loss_fake

        self.sum_g = tf.summary.scalar('G_loss', self.G_loss)
        self.sum_d = tf.summary.scalar('D_loss', self.D_loss)

        # variables
        self.G_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        self.D_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

    	# learning rate
        self.sum_lr = tf.summary.scalar('sum_lr', self.lr)

        # this is better than Adam
        # self.G_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.G_vars)
        
        # adam for generator, rmsprop for discriminator
        self.G_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.G_vars)
        # self.D_opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr+2e-5).minimize(self.D_loss, var_list=self.D_vars)
        self.D_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.D_vars)


        # saver
        self.saver = tf.train.Saver()

        if self.load_checkpoint is not None:
        	# ckpt = tf.train.latest_checkpoint(self.load_checkpoint)
        	ckpt = self.load_checkpoint + 'model.ckpt-100000'
        	print ('reloading checkpoint from %s' % ckpt)
        	self.saver.restore(self.sess, ckpt)
        	self.start = 1
        else:
	        tf.global_variables_initializer().run(session=self.sess)

    def save_model(self, step=None):

        self.saver.save(self.sess, os.path.join(self.model_path, "model.ckpt"), global_step=step)

    def save_result(self, step):

    	feed_dict = {
    		self.noise: self.test_vec,
    		self.keep_prob: 1.0,
    		self.is_training: True
    	}

    	pics = self.sess.run(self.G, feed_dict=feed_dict)

    	fig = plt.figure()

    	for i, p in enumerate(pics):

    		# normalize p
    		for j in range(3):

    			tmp = p[:, :, j].reshape(-1)
    			tmp = (tmp - tmp.min()) / (tmp.max() - tmp.min())
    			p[:, :, j] = tmp.reshape((pics.shape[1], pics.shape[2]))

    		fig.add_subplot(5, 5, i+1)
    		plt.imshow(p)

    	plt.savefig(os.path.join(self.vis_path, '%04d_updates' % step))

    	print ('- Saving %04d_updates to {}.'.format(self.vis_path) % step)

    def train(self):

        # remove previous tensorboard files
        shit = [f for f in os.listdir(self.tensorboard_path)]
        for s in shit:
        	os.remove(os.path.join(self.tensorboard_path, s))

        img_gen = self.image_generator()
        writer  = tf.summary.FileWriter(self.tensorboard_path, graph=self.sess.graph)
        start_time = time.time()

        for step in range(self.start, self.steps):

        	# decay learning rate
        	learning_rate = self.decay_lr(step, 5e3, 0.9)

        	self.batch = img_gen.__next__()
        	self.z     = self.random_noise()

        	feed_dict_d = {
        		self.image: self.batch,
       			self.noise: self.z,
       			self.keep_prob: self.dropout,
       			self.is_training: True,
       			self.lr: learning_rate
       		}

       		feed_dict_g = {
       			self.noise: self.z,
       			self.keep_prob: self.dropout,
       			self.is_training: True,
       			self.lr: learning_rate
       		}

       		# update D network
       		things = [self.D_opt, self.sum_d, self.D_loss_real, self.D_loss_fake, self.D_loss, self.sum_lr, self.sum_real_out, self.sum_fake_out]
	       	_, summary_d, d_loss_real, d_loss_fake, d_loss, summary_lr, summary_real_out, summary_fake_out = self.sess.run(things, feed_dict=feed_dict_d)
	       	
	       	writer.add_summary(summary_d, step)
	       	writer.add_summary(summary_lr, step)
	       	writer.add_summary(summary_real_out, step)
	       	writer.add_summary(summary_fake_out, step)

       		# update G network
       		# self.sess.run([self.G_opt, self.G_loss], feed_dict=feed_dict_g)
       		# self.sess.run([self.G_opt, self.G_loss], feed_dict=feed_dict_g)

       		# update G network twice
	       	things = [self.G_opt, self.sum_g, self.G_loss, self.sum_img, self.sum_gen_out]
	       	_, summary, g_loss, image, summary_gen_out = self.sess.run(things, feed_dict=feed_dict_g)
	       	
	       	writer.add_summary(summary, step)
	       	writer.add_summary(image, step)
	       	writer.add_summary(summary_gen_out, step)


       		print ("Step: [%6d/%6d] time: %4.4fs, lr: %.3e, d_loss_real: %.4f, d_loss_fake: %.4f, d_loss: %.4f, g_loss: %.4f" \
          			% (step, self.steps-1, time.time() - start_time, learning_rate, d_loss_real, d_loss_fake, d_loss, g_loss))

        	# if step != 0 and step % 2000 == 0:
        print ('End of training, saving model...')
        self.save_model(step=step)

    def ReadData(self):

        print ('Reading Data...', end='')

        self.train_pics = list()

        dirs = os.listdir(self.data_path)

        for d in dirs:

            try:
                pics = os.listdir(os.path.join(self.data_path, d))
            except:
                continue

            for p in pics:

                self.train_pics.append(os.path.join(self.data_path, d, p))

        print ('{} training data.'.format(len(self.train_pics)))

    def image_generator(self):

    	batch_count = len(self.train_pics) // self.batch_size

    	while True:

    		random.shuffle(self.train_pics)

    		for batch in range(batch_count):

    			img = list()
    			
    			for i in range(self.batch_size):

    				pth = self.train_pics[batch*self.batch_size+i]
    				img.append(np.array(imread(pth)) / 255 * 2 - 1) # normalize to (-1, 1)
    		
    			yield np.array(img)

    def decay_lr(self, step, cycle, decay_rate):

    	if step == self.start:

    		self.current_lr = self.base_learning_rate
    		return self.current_lr
      
    	elif step % cycle == 0:

    		self.current_lr *= decay_rate

    	return self.current_lr

if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
	# test = DCGAN(sess)
	# tf.global_variables_initializer().run(session=sess)
	# test.save_model()

	test = DCGAN(sess, reload_model='../model/step-110000')
