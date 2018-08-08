import tensorflow as tf
import numpy as np
import matplotlib, time, os, random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.misc import imread



class DCGAN(object):
    """docstring for DCGAN"""
    def __init__(self, sess, batch_size=128, latent_dim=100, learning_rate=1e-4, steps=6000, drop_rate=0.5,
    			 interval=20, model_path='../model/', vis_path='../result/', data_path='../anime-faces', reload_model=None):
        self.sess = sess
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.steps = steps
        self.momentum = 0.99
        self.interval = interval # interval to evaluate losses
        self.dropout = 1 - drop_rate
        self.model_path = model_path
        self.vis_path = vis_path
        self.data_path = data_path
        self.test_vec = np.random.uniform(0, 1, [25, self.latent_dim]).astype(np.float32) # used for visualization during training

        if not os.path.isdir(self.model_path):
        	os.mkdir(self.model_path)
        if not os.path.isdir(self.vis_path):
        	os.mkdir(self.vis_path)

        self.ReadData()
        if reload_model is None:
            self.CreateModel()
        else:
            self.load_model(reload_model)

    def load_model(self, model):

        meta_file = [f for f in os.listdir(model) if f.endswith('meta')]

        assert len(meta_file) == 0, 'There should only be one meta file in the model directory.'

        meta_file = meta_file[0]

        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(self.sess, tf.train.latest_checkpoint(model))

    def random_noise(self):

    	return np.random.uniform(0, 1, [self.batch_size, self.latent_dim]).astype(np.float32)

    def deconv2d(self, inputs, filters, kernel_size, alpha=0.2, padding='same', sigmoid=False):

        def lrelu(x):
            return tf.nn.leaky_relu(x, alpha=alpha)

        if not sigmoid:
            activation = lrelu
        else:
            activation = tf.nn.sigmoid

        tmp = tf.layers.conv2d_transpose(
                inputs=inputs,
                filters=filters,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.02),
                kernel_size=kernel_size,
                strides=2,
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

    def binary_cross_entropy(self, label, z):
    	eps = 1e-12
    	return (-(label * tf.log(z + eps) + (1. - label) * tf.log(1. - z + eps)))

    def generator(self, z, keep_prob, is_training):
        with tf.variable_scope('generator') as scope:
            print ('==================================================================')
            print ('GENERATOR\n     {:40} {}'.format('Layer', 'Shape'))
            # shape 1*1*100
            x = z
            print ('{:40} {}'.format(x.name, x.shape))
            x = tf.layers.dense(x, units=self.latent_dim, activation=tf.nn.relu)
            x = tf.reshape(z, shape=[-1, 1, 1, self.latent_dim]) # try different shape
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 3*3*1024
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum)
            x = self.deconv2d(x, 1024, [3, 3], padding='valid')
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 6*6*512
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum)
            x = self.deconv2d(x,  512, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 12*12*256
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum)
            x = self.deconv2d(x,  512, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 24*24*128
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum)
            x = self.deconv2d(x,  256, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 48*48*64
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum)
            x = self.deconv2d(x,  128, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 96*96*3
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum)
            x = self.deconv2d(x,    3, [3, 3], sigmoid=True)
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}\n'.format(x.name, x.shape))

        return x

    def discriminator(self, image, keep_prob, reuse=False):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()

            print ('=====================================================')
            print ('DISCRIMINATOR\n     {:40} {}'.format('Layer', 'Shape'))
            # shape 96*96*3
            x = image
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 32*32*64
            # x = tf.contrib.layers.batch_norm(x)
            x = self.conv2d(x,  64, [3, 3], strides=3)
            x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 16*16*128
            # x = tf.contrib.layers.batch_norm(x)
            x = self.conv2d(x, 128, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 8*8*256
            # x = tf.contrib.layers.batch_norm(x)
            x = self.conv2d(x, 256, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 4*4*512
            # x = tf.contrib.layers.batch_norm(x)
            x = self.conv2d(x, 256, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 2*2*1024
            # x = tf.contrib.layers.batch_norm(x)
            x = self.conv2d(x, 512, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:40} {}'.format(x.name, x.shape))
            # shape 1*1*1
            # x = tf.contrib.layers.batch_norm(x)
            x = self.conv2d(x,    1, [3, 3])
            print ('{:40} {}'.format(x.name, x.shape))
            # reshape
            # x = tf.reshape(x, [-1])
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
            print ('{:40} {}\n'.format(x.name, x.shape))

        return x

    def CreateModel(self):

    	# place holders
        self.image = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        self.noise = tf.placeholder(tf.float32, [None, self.latent_dim], name='noise')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')

        # models
        self.G      = self.generator(self.noise, self.dropout, self.is_training)
        self.D_real = self.discriminator(self.image, self.dropout)
        self.D_fake = self.discriminator(self.G, self.dropout, reuse=True)

        self.sum_img = tf.summary.image('generated', self.G, max_outputs=4)

        # define loss
        with tf.name_scope('Loss'):
	        self.G_loss = tf.reduce_mean(self.binary_cross_entropy(tf.ones_like(self.D_fake), self.D_fake))
	        self.D_loss_real = self.binary_cross_entropy(tf.ones_like(self.D_real), self.D_real)
	        self.D_loss_fake = self.binary_cross_entropy(tf.zeros_like(self.D_fake), self.D_fake)
	        self.D_loss = tf.reduce_mean(0.5 * (self.D_loss_real + self.D_loss_fake))

        self.sum_g = tf.summary.scalar('G_loss', self.G_loss)
        self.sum_d = tf.summary.scalar('D_loss', self.D_loss)

        # variables
        self.G_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        self.D_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

        # saver
        self.saver = tf.train.Saver()

    def save_model(self, step=None):

        if step is None:
            if not os.path.isdir(os.path.join(self.model_path, 'best')):
                os.mkdir(os.path.join(self.model_path, 'best'))
            self.saver.save(self.sess, os.path.join(self.model_path, 'best', "model.ckpt"))
        else:
            if not os.path.isdir(os.path.join(self.model_path, 'step-'+str(step))):
                os.mkdir(os.path.join(self.model_path, 'step-'+str(step)))
            self.saver.save(self.sess, os.path.join(self.model_path, 'step-'+str(step)), "model.ckpt")

    def save_result(self, step):

    	pics = self.sess.run(self.G,
    						feed_dict={
    							self.noise: self.test_vec,
    							self.keep_prob: 1.0,
    							self.is_training: False
    						})

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

        self.G_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.G_loss, var_list=self.G_vars)
        self.D_opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.D_loss, var_list=self.D_vars)

        img_gen = self.image_generator()
        total_loss = 1000
        start_time = time.time()
        writer = tf.summary.FileWriter("../TensorBoard/", graph=self.sess.graph)
        tf.global_variables_initializer().run(session=self.sess)

        # list to contain g_loss
        losses = list()

        for step in range(self.steps):

        	self.batch = img_gen.__next__()
        	self.z     = self.random_noise()

        	feed_dict_d = {
        		self.image: self.batch,
       			self.noise: self.z,
       			self.keep_prob: self.dropout,
       			self.is_training: True
       		}

       		feed_dict_g = {
       			self.noise: self.z,
       			self.keep_prob: self.dropout,
       			self.is_training: True
       		}

       		# update D network
       		_, summary, d_loss_real, d_loss_fake, d_loss = self.sess.run([self.D_opt, self.sum_d, self.D_loss_real, self.D_loss_fake, self.D_loss],
       													feed_dict=feed_dict_d)
       		writer.add_summary(summary, step)

       		# update G network
       		_, g_loss = self.sess.run([self.G_opt, self.G_loss],
       								feed_dict=feed_dict_g)

       		# update G network
       		_, summary, g_loss, image = self.sess.run([self.G_opt, self.sum_g, self.G_loss, self.sum_img],
       								feed_dict=feed_dict_g)
       		writer.add_summary(summary, step)
       		writer.add_summary(image, step)



       		d_loss_real = d_loss_real.mean()
       		d_loss_fake = d_loss_fake.mean()

       		print ("Step: [%6d/%6d] time: %4.4fs, d_loss_real: %.8f, d_loss_fake: %.8f, d_loss: %.8f, g_loss: %.8f" \
          			% (step, self.steps, time.time() - start_time, d_loss_real, d_loss_fake, d_loss, g_loss))

       		# store g_loss
       		losses.append(g_loss)

        	# evaluate training process, parameters identical to training process
        	if step != 0 and step % self.interval == 0:

        		g_loss = np.array(losses).mean()
        		losses = list()

        		print ('-------------------------------------------------------')
        		print ('- Best g_loss: %.4f' % total_loss, 'Current g_loss: %.4f' % g_loss)

        		if total_loss > g_loss:
        			total_loss = g_loss

        			if step >= 10000:
        				self.save_model()
        				print ('- Saving model to {}'.format(self.model_path))
        		
        				self.save_result(step)
        		print ('-------------------------------------------------------')

        	# save model every 1000 step
        	if step != 0 and step > 50000 and step % 1000 == 0:
        		self.save_model(step=step)

    def ReadData(self):

        print ('Reading Data...')

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
    				img.append(np.array(imread(pth)) / 255)
    		
    			yield np.array(img)

    def test(self):

    	print (self.random_noise())
    	print (self.random_noise())
    	print (self.random_noise())

        # z = tf.placeholder(tf.float32, shape=[None, 1, 1, 100], name='random_input')
        # self.generator(z)

        # image = tf.placeholder(tf.float32, shape=[None, 96, 96, 3], name='image')
        # self.discriminator(image)

        # G = self.image_generator()

        # batch = G.__next__()

        # print (batch[0])



if __name__ == '__main__':
    # fuck = DCGAN(tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)))
    fuck.test()
    # fuck.train()
