from model import DCGAN

import tensorflow as tf


# specify gpu
import os, sys, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# allow memory growth
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
gpu_options = tf.GPUOptions(allow_growth=True)



if __name__ == '__main__':
	
	gan = DCGAN(
		tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)),
		drop_rate=0.4,
		interval=50,
		steps=100000,
		learning_rate=3e-4,
		batch_size=128,
		)

	gan.train()