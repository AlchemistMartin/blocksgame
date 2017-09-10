# -*- coding: utf8 -*-
import tensorflow as tf
import numpy as np
import game


def get_move(state):
	all_plane = state['overall']
	cur_plane = state['cur_plane']
	nxt_plane = state['nxt_plane']
	ali_plane = state['ali_plane']
	pre_plane = state['pre_plane']
	cur_box = state['cur_box']
	nxt_box = state['nxt_box']
	ali_box = state['ali_box']
	pre_box = state['pre_box']
	cur_x = np.array([all_plane, cur_plane, nxt_plane, ali_plane, pre_plane] + cur_box)
	x_array = np.array([cur_x])
	x_ = x_array.reshape(x_array.shape[0], x_array.shape[2], x_array.shape[3], x_array.shape[1])

	sess = tf.Session()
	saver = tf.train.import_meta_graph('test-model.meta')
	saver.restore(sess, tf.train.latest_checkpoint('./'))
	graph = tf.get_default_graph()
	y_pred = graph.get_tensor_by_name("y_pred:0")
	x = graph.get_tensor_by_name("x:0")
	result = sess.run(y_pred, feed_dict={x: x_})
	return result

state = game.ini_game()
move = get_move(state)
print(move.reshape(20, 20) > 0.1)

