# -*- coding: utf8 -*-
from glob import glob
import json
import game
import numpy as np


class DataSet(object):

	def __init__(self, images, labels):
		self._num_examples = images.shape[0]

		self._images = images
		self._labels = labels
		self._epochs_done = 0
		self._index_in_epoch = 0

	@property
	def images(self):
		return self._images

	@property
	def labels(self):
		return self._labels

	@property
	def num_examples(self):
		return self._num_examples

	@property
	def epochs_done(self):
		return self._epochs_done

	def next_batch(self, batch_size):
		"""Return the next `batch_size` examples from this data set."""
		start = self._index_in_epoch
		self._index_in_epoch += batch_size

		if self._index_in_epoch > self._num_examples:
		# After each epoch we update this
			self._epochs_done += 1
			start = 0
			self._index_in_epoch = batch_size
		assert batch_size <= self._num_examples
		end = self._index_in_epoch

		return self._images[start:end], self._labels[start:end]


def txt_pb_read(txt_dir):
	txt_dir += '*.txt'
	x_list = []
	y_list = []
	for r in glob(txt_dir):
		with open(r) as f:
			pb = f.read()
		pb_json = json.loads(pb[10:-1])
		state = game.ini_game()
		user = 0
		for msg in pb_json:
			move = np.zeros((20, 20))
			if 'chessman' in msg['msg_data'].keys() and 'squareness' in msg['msg_data']['chessman'].keys():
				grids = msg['msg_data']['chessman']['squareness']
				for grid in grids:
					move[(grid['x'], grid['y'])] = 1
			if 'player_id' in msg['msg_data'].keys():
				user = msg['msg_data']['player_id'] - 1
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
			cur_y = move
			x_list.append(cur_x)
			y_list.append(cur_y)
			if not np.array_equal(move, np.zeros((20, 20))):
				state = game.update_game(state, user, move)
	x_array = np.array(x_list)
	y_array = np.array(y_list)
	x_sets = x_array.reshape(x_array.shape[0], x_array.shape[2], x_array.shape[3], x_array.shape[1])
	y_sets = y_array.reshape(y_array.shape[0], y_array.shape[1] * y_array.shape[2])
	return DataSet(x_sets, y_sets)

# data = txt_pb_read('/Volumes/M2/CurrentEdit/playbook/gamebook/replay/Blokus10/')
# x, y = data.next_batch(10)
# print(x.shape)
# print(y.shape)