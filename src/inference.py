import tensorflow as tf
import numpy as np
from scipy.misc import imsave

# specify gpu
import os, sys, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# allow memory growth
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
gpu_options = tf.GPUOptions(allow_growth=True)


# load model
ckpt = "model.ckpt-100000"
meta_file = ckpt + ".meta"
meta_file = os.path.join(sys.argv[1], meta_file)


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

	saver = tf.train.import_meta_graph(meta_file)
	saver.restore(sess, os.path.join(sys.argv[1], ckpt))

	gen_image = tf.get_default_graph().get_tensor_by_name('generator/conv2d_transpose_5/Tanh:0')
	noise     = tf.get_default_graph().get_tensor_by_name('noise:0')
	keep_prob = tf.get_default_graph().get_tensor_by_name('keep_prob:0')
	is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')

	print (noise)
	print (gen_image)
	print (keep_prob)
	print (is_training)



	images = sess.run(gen_image,
				feed_dict={
					noise: np.random.normal(-1, 1, [50, 100]).astype(np.float32),
					keep_prob: 1.0,
					is_training: True
				})

	print (images.shape)


	for i, img in enumerate(images):

		img = img.reshape(-1)
		img = (img - img.min()) / (img.max() - img.min())
		img = img.reshape((96, 96 ,3))

		imsave('../result/%04d.jpg' % i, img)










