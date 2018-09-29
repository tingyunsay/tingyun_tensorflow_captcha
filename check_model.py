#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='2'
from train import *
from gen_image import *

def crack_captcha(captcha_image):
	#构造图结构
	output = crack_captcha_cnn()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		#saver.restore(sess, tf.train.latest_checkpoint('.'))
		#载入模型
		saver.restore(sess, "./model/tingyun_crack_capcha_model.ckpt")

		predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
		text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})

		text = text_list[0].tolist()
		vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
		i = 0
		for n in text:
				vector[i*CHAR_SET_LEN + n] = 1
				i += 1
		return vec2text(vector)

text, image = gen_captcha_text_and_image()
image = convert2gray(image)
image = image.flatten() / 255
predict_text = crack_captcha(image)
print("正确: {}  预测: {}".format(text, predict_text))
