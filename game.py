# -*- coding: utf8 -*-
import numpy as np
import json
import tensorflow as tf


def pad20(tar_arr):
	return np.pad(tar_arr, pad_width=((0, 20-tar_arr.shape[0]), (0, 20-tar_arr.shape[1])), mode='constant', constant_values=0)


def ini_pieces():
	with open("pieces.json") as json_file:
		json_c = json_file.read()
	pieces = json.loads(json_c)
	ori_pieces = {}
	all_pieces = {}
	for key in pieces.keys():
		shape = int(key)
		ori_move = np.array(pieces[key])
		ori_lr = np.fliplr(ori_move)
		ori_ud = np.flipud(ori_move)
		rot_90 = np.rot90(ori_move, 1)
		rot_90_lr = np.fliplr(rot_90)
		rot_90_ud = np.flipud(rot_90)
		rot_180 = np.rot90(ori_move, 2)
		rot_180_lr = np.fliplr(rot_180)
		rot_180_ud = np.flipud(rot_180)
		rot_270 = np.rot90(ori_move, 3)
		rot_270_lr = np.fliplr(rot_270)
		rot_270_ud = np.flipud(rot_270)
		possible_shapes = np.asarray([pad20(ori_move), pad20(ori_lr), pad20(ori_ud),
						pad20(rot_90), pad20(rot_90_lr), pad20(rot_90_ud),
						pad20(rot_180), pad20(rot_180_lr), pad20(rot_180_ud),
						pad20(rot_270), pad20(rot_270_lr), pad20(rot_270_ud)])
		uniques = []
		for arr in possible_shapes:
			if not any(np.array_equal(arr, unique_arr) for unique_arr in uniques):
				uniques.append(arr)
		ori_pieces[shape] = pad20(ori_move)
		all_pieces[shape] = uniques
	return ori_pieces, all_pieces
ORI_PIECES, ALL_PIECES = ini_pieces()


def ini_box():
	box = []
	for i in range(0, 21):
		box.append(ORI_PIECES[i])
	return box


def get_piece(move):
	x_ = np.nonzero(move)[0].min()
	y_ = np.nonzero(move)[1].min()
	move = pad20(move[x_:, y_:])
	for shape in ALL_PIECES.keys():
		for unique in ALL_PIECES[shape]:
			if np.array_equal(move, unique):
				return shape


def move2str(move):
	alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
	x1 = np.nonzero(move)[0]
	y1 = np.nonzero(move)[1]
	grids = []
	for i in range(0, x1.__len__()):
		fig = 19 - x1[i]
		al = alpha[y1[i]]
		grids.append(al + str(fig))
	return ','.join(grids)


def str2move(move_str):
	alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
	move = np.zeros((20, 20))
	if move_str != '':
		moves = move_str.split(',')
		for grid in moves:
			x_ = 19 - int(grid[1:])
			y_ = alpha.index(grid[0])
			move[x_, y_] = 1
	return move


def get_neis(move):
	xs = np.nonzero(move)[0]
	ys = np.nonzero(move)[1]
	grids = []
	reg_ori = []
	piv_ori = []
	for i in range(0, xs.__len__()):
		x_ = xs[i]
		y_ = ys[i]
		grids.append((x_, y_))
		reg_ori.extend([(x_-1, y_), (x_+1, y_), (x_, y_-1), (x_, y_+1)])
		piv_ori.extend([(x_-1, y_-1), (x_-1, y_+1), (x_+1, y_-1), (x_+1, y_+1)])
	reg_ori = list(set(reg_ori))
	piv_ori = list(set(piv_ori))
	reg_list = [(), ()]
	piv_avoid = []
	piv_list = [(), ()]
	for ele in reg_ori:
		if ele[0] in range(0, 5) and ele[1] in range(0, 5) and ele not in grids:
			reg_list[0] += (ele[0],)
			reg_list[1] += (ele[1],)
			piv_avoid.append((ele[0], ele[1]))
	for ele in piv_ori:
		if ele[0] in range(0, 5) and ele[1] in range(0, 5) and ele not in grids and ele not in piv_avoid:
			piv_list[0] += (ele[0],)
			piv_list[1] += (ele[1],)
	neis = {'reg': reg_list, 'piv': piv_list}
	return neis


def legit(state, move):
	legal = {'val': True, 'msg': ''}
	# 判断全0的的落子
	if move.sum() == 0:
		legal['val'] = False
		legal['msg'] += 'Empty Move;'
		return legal
	# 是否顶格落子，如果是第一步，需要判断是否在各自顶格
	starts = {0: (0, 0), 1: (0, 19), 2: (19, 19), 3: (19, 0)}
	user = state['cur_user']
	if state['cur_plane'].sum() == 0:
		if move[starts[user]] != 0:
			legal['val'] = False
			legal['msg'] += 'Started Wrong;'
	else:
		neis = get_neis(move)
		if state['cur_plane'][neis['reg']].sum() > 0 or state['cur_plane'][neis['piv']].sum() < 1:
			legal['val'] = False
			legal['msg'] += 'Wrong Position；'
	# 是不是盒中的子
	shape = get_piece(move)
	if shape:
		box = state['cur_box']
		if box[shape].sum() == 0:
			legal['val'] = False
			legal['msg'] += 'Not in box；'
	# 有没有地方放
	all_plane = state['overall']
	if (all_plane + move).max() > 1:
		legal['val'] = False
		legal['msg'] += 'No Room；'
	return legal


def ini_game():
	state = dict()
	state['cur_user'] = 0
	state['planes'] = [np.zeros((20, 20)), np.zeros((20, 20)), np.zeros((20, 20)), np.zeros((20, 20))]
	state['overall'] = np.zeros((20, 20))
	state['cur_plane'] = np.zeros((20, 20))
	state['ali_plane'] = np.zeros((20, 20))
	state['nxt_plane'] = np.zeros((20, 20))
	state['pre_plane'] = np.zeros((20, 20))
	state['boxes'] = [ini_box(), ini_box(), ini_box(), ini_box()]
	state['cur_box'] = ini_box()
	state['ali_box'] = ini_box()
	state['nxt_box'] = ini_box()
	state['pre_box'] = ini_box()
	return state


def update_game(state, user, move):
	new_state = state.copy()
	user_dict = {0: [1, 2, 3, 0], 1: [2, 3, 0, 1], 2: [3, 0, 1, 2], 3: [0, 1, 2, 3]}
	# move有效时加move，否则只更新用户
	new_state['cur_user'] = (user < 3) * (user + 1)
	new_state['overall'] = state['overall'] + move
	new_state['planes'][user] += move
	shape = get_piece(move)
	new_state['boxes'][user][shape] = np.zeros((20, 20))
	new_state['cur_plane'] = new_state['planes'][user_dict[user][0]]
	new_state['nxt_plane'] = new_state['planes'][user_dict[user][1]]
	new_state['ali_plane'] = new_state['planes'][user_dict[user][2]]
	new_state['pre_plane'] = new_state['planes'][user_dict[user][3]]
	new_state['cur_box'] = new_state['boxes'][user_dict[user][0]]
	new_state['nxt_box'] = new_state['boxes'][user_dict[user][1]]
	new_state['ali_box'] = new_state['boxes'][user_dict[user][2]]
	new_state['pre_box'] = new_state['boxes'][user_dict[user][3]]
	return new_state


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
	result = (result > 0.5).reshape(20, 20)
	return result


def score_board(state):
	score0 = state['planes'][0].sum()
	score1 = state['planes'][0].sum()
	score2 = state['planes'][0].sum()
	score3 = state['planes'][0].sum()
	score02 = score0 + score2
	score13 = score1 + score3
	winner = '02'
	lead = score02 - score13
	if score13 > score02:
		winner = '13'
		lead *= -1
	msg = "team02: {0}; team13: {1}; Players: {2}; {3}; {4}; {5}, team{6} lead by {7}"
	msg.format(score02, score13, score0, score1, score2, score3, winner, lead)
	result = {'msg': msg}
	return result


def game_on():
	state = ini_game()
	teams = {0: 'ai', 1: 'ptb', 2: 'ai', 3: 'ptb'}
	skip = 0
	while skip < 4:
		cur_user = state['cur_user']
		if teams[cur_user] == 'ai':
			move_tmp = get_move(state)
			legal = legit(state, move_tmp)
			print move_tmp
			print legal['msg']
			move_tmp_str = move2str(move_tmp) + '(y/n)?:'
			review = raw_input(move_tmp_str)
			if review == '':
				move = np.zeros((20, 20))
				skip += 1
			elif review == 'y':
				move = move_tmp
				skip = 0
			else:
				move = str2move(review)
				skip = 0
			state = update_game(state, cur_user, move)
		else:
			move_str = raw_input("ptb move str:")
			if move_str == '':
				move = np.zeros((20, 20))
				skip += 1
			else:
				move = str2move(move_str)
				print(move)
				skip = 0
			state = update_game(state, cur_user, move)
		print(score_board(state)['msg'])
	print 'game over'

game_on()

# with open('test.json') as tef:
# 	te = tef.read()
# print json.loads(te)
