# -*- coding: utf8 -*-
import numpy as np
import json


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
		fig = 20 - x1[i]
		al = alpha[y1[i]]
		grids.append(al + str(fig))
	return grids


def str2move(move_str):
	alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
	move = np.zeros((20, 20))
	if move_str != '':
		moves = move_str.split(',')
		for grid in moves:
			x_ = 20 - int(grid[1:])
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





# with open('test.json') as tef:
# 	te = tef.read()
# print json.loads(te)
