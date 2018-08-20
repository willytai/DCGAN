import tensorflow as tf
import numpy as np
import matplotlib, time, os, random
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.misc import imread



class DCGAN(object):
    """docstring for DCGAN"""
    def __init__(self, sess, batch_size=128, latent_dim=100, learning_rate=5e-5, steps=6000, drop_rate=0.5,
    	interval=20, model_path='../model/', vis_path='../result/', data_path='../anime-faces', reload_model=None,
    	tensorboard_path='../TensorBoard'):

        self.sess = sess
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.base_learning_rate = learning_rate
        self.steps = steps
        self.momentum = 0.99
        self.interval = interval # interval to evaluate losses
        self.dropout = 1 - drop_rate
        self.model_path = model_path
        self.vis_path = vis_path
        self.data_path = data_path
        self.tensorboard_path = tensorboard_path
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

        if not os.path.isdir(self.model_path):
        	os.mkdir(self.model_path)

        if not os.path.isdir(self.tensorboard_path):
        	os.mkdir(self.tensorboard_path)

    def load_model(self, model):

        meta_file = [f for f in os.listdir(model) if f.endswith('meta')]

        assert len(meta_file) == 1, 'There should only be one meta file in the model directory.'

        meta_file = os.path.join(model, meta_file[0])

        tf.reset_default_graph()
        saver = tf.train.import_meta_graph(meta_file)
        saver.restore(self.sess, tf.train.latest_checkpoint(model))

        # set parameters
        self.image = tf.get_default_graph().get_tensor_by_name('image:0')
        self.noise = tf.get_default_graph().get_tensor_by_name('noise:0')
        self.keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
        self.is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')
        self.lr = tf.get_default_graph().get_tensor_by_name('learning_rate:0')

        self.G = tf.get_default_graph().get_tensor_by_name('generator/conv2d_transpose_5/Sigmoid:0')
        self.D_real = tf.get_default_graph().get_tensor_by_name('discriminator/dense/Sigmoid:0')
        self.D_fake = tf.get_default_graph().get_tensor_by_name('discriminator_1/dense/Sigmoid:0')

        self.sum_img = tf.summary.image('generated', self.G, max_outputs=4)

        self.G_loss = tf.get_default_graph().get_tensor_by_name('Loss/Mean:0')
        self.D_loss_real = tf.get_default_graph().get_tensor_by_name('Loss/Neg_1:0')
        self.D_loss_fake = tf.get_default_graph().get_tensor_by_name('Loss/Neg_2:0')
        self.D_loss = tf.get_default_graph().get_tensor_by_name('Loss/Mean_1:0')

        self.sum_g = tf.get_default_graph().get_tensor_by_name('G_loss:0')
        self.sum_d = tf.get_default_graph().get_tensor_by_name('D_loss:0')

        self.G_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
        self.D_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

        self.saver = tf.train.Saver()

    def random_noise(self):

    	return np.random.uniform(0, 1, [self.batch_size, self.latent_dim]).astype(np.float32)

    def deconv2d(self, inputs, filters, kernel_size, alpha=0.2, padding='same', sigmoid=False, strides=2):

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

    def binary_cross_entropy(self, label, z):
    	eps = 1e-12
    	return (-(label * tf.log(z + eps) + (1. - label) * tf.log(1. - z + eps)))

    def generator(self, z, keep_prob, is_training):
        with tf.variable_scope('generator') as scope:
            print ('=======================================================================')
            print ('GENERATOR\n     {:50} {}'.format('Layer', 'Shape'))
            # shape 100
            x = z
            print ('{:50} {}'.format(x.name, x.shape))
            # project and reshape 1*1*1024
            x = tf.layers.dense(x, units=1024, activation=tf.nn.relu, 
            	kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.02))
            x = tf.reshape(x, shape=[-1, 1, 1, 1024])
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 3*3*1024
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x, 1024, [3, 3], padding='valid')
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 6*6*512
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,  512, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 12*12*256
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,  256, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 24*24*128
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,  128, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 48*48*64
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,  64, [3, 3])
            # x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 96*96*3
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.deconv2d(x,    3, [3, 3], sigmoid=True)
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
            x = self.conv2d(x,  512, [3, 3], strides=3)
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 16*16*128
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,  256, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 8*8*256
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,  128, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 4*4*512
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,   64, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 2*2*1024
            x = tf.contrib.layers.batch_norm(x, is_training=is_training, decay=self.momentum, scale=True)
            x = self.conv2d(x,   32, [3, 3])
            x = tf.layers.dropout(x, keep_prob)
            print ('{:50} {}'.format(x.name, x.shape))
            # shape 1*1*1
            # x = tf.contrib.layers.batch_norm(x)
            x = self.conv2d(x,    1, [3, 3])
            print ('{:50} {}'.format(x.name, x.shape))
            # reshape
            x = tf.layers.flatten(x)
            x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
            print ('{:50} {}\n'.format(x.name, x.shape))

        return x

    def CreateModel(self):

    	# placeholders
        self.image = tf.placeholder(tf.float32, [None, 96, 96, 3], name='image')
        self.noise = tf.placeholder(tf.float32, [None, self.latent_dim], name='noise')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        # models
        self.G      = self.generator(self.noise, self.dropout, self.is_training)
        self.D_real = self.discriminator(self.image, self.dropout, self.is_training)
        self.D_fake = self.discriminator(self.G, self.dropout, self.is_training, reuse=True)

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
            self.saver.save(self.sess, os.path.join(self.model_path, 'step-'+str(step), "model.ckpt"))

    def save_result(self, step):

    	feed_dict = {
    		self.noise: self.test_vec,
    		self.keep_prob: 1.0,
    		self.is_training: False
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

    	# learning rate
        self.sum_lr = tf.summary.scalar('sum_lr', self.lr)

        # this is better than Adam
        self.G_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.G_loss, var_list=self.G_vars)
        self.D_opt = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(self.D_loss, var_list=self.D_vars)

        img_gen = self.image_generator()
        writer  = tf.summary.FileWriter(self.tensorboard_path, graph=self.sess.graph)
        total_loss = 1000
        start_time = time.time()

        tf.global_variables_initializer().run(session=self.sess)

        # list to contain g_loss
        losses = list()

        for step in range(self.steps):

        	# decay learning rate
        	learning_rate = self.decay_lr(step, 5e3, 0.9)
        	# learning_rate = tf.train.exponential_decay(self.base_learning_rate, step, 5e3, 0.9, staircase=True)

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
       		things = [self.D_opt, self.sum_d, self.D_loss_real, self.D_loss_fake, self.D_loss, self.sum_lr]
       		_, summary_d, d_loss_real, d_loss_fake, d_loss, summary_lr = self.sess.run(things, feed_dict=feed_dict_d)
       		
       		writer.add_summary(summary_d, step)
       		writer.add_summary(summary_lr, step)

       		# update G network
       		_, g_loss = self.sess.run([self.G_opt, self.G_loss], feed_dict=feed_dict_g)

       		# update G network twice
       		things = [self.G_opt, self.sum_g, self.G_loss, self.sum_img]
       		_, summary, g_loss, image = self.sess.run(things, feed_dict=feed_dict_g)
       		
       		writer.add_summary(summary, step)
       		writer.add_summary(image, step)

       		d_loss_real = d_loss_real.mean()
       		d_loss_fake = d_loss_fake.mean()

       		print ("Step: [%6d/%6d] time: %4.4fs, lr: %.3e, d_loss_real: %.4f, d_loss_fake: %.4f, d_loss: %.4f, g_loss: %.4f" \
          			% (step, self.steps, time.time() - start_time, learning_rate, d_loss_real, d_loss_fake, d_loss, g_loss))

       		# store g_loss
       		losses.append(g_loss)

        	# evaluate training process every specified interval
        	if step != 0 and step % self.interval == 0:

        		g_loss = np.array(losses).mean()
        		losses = list()

        		print ('-------------------------------------------------------')
        		print ('- Best g_loss: %.4f' % total_loss, 'Current g_loss: %.4f' % g_loss)
        		print ('- Learning rate: %.8f' % learning_rate)

        		if total_loss > g_loss and step >= 10000:
        			total_loss = g_loss

        			if step >= 10000:
        				self.save_model()
        				print ('- Saving model to {}'.format(self.model_path))
        		
        				self.save_result(step)
        		print ('-------------------------------------------------------')

        	# save model every 1000 step after updating 50000 times
        	if step != 0 and step >= 50000 and step % 10000 == 0:
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
    				img.append(np.array(imread(pth)) / 255)
    		
    			yield np.array(img)

    def decay_lr(self, step, cycle, decay_rate):

    	if step == 0:

    		self.current_lr = self.base_learning_rate
    		return self.current_lr

    	elif step % cycle == 0:

    		self.current_lr *= decay_rate

    	return self.current_lr

if __name__ == '__main__':
	sess = tf.Session()
	# test = DCGAN(sess)
	# tf.global_variables_initializer().run(session=sess)
	# test.save_model()

	test = DCGAN(sess, reload_model='../model/best')
