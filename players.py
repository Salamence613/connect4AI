import random
import sys
import time
from copy import deepcopy

import numpy as np
import pygame
import math

class connect4Player(object):
	def __init__(self, position, seed=0):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)

	def play(self, env, move):
		move = [-1]

class human(connect4Player):

	def play(self, env, move):
		move[:] = [int(input('Select next move: '))]
		while True:
			if int(move[0]) >= 0 and int(move[0]) <= 6 and env.topPosition[int(move[0])] >= 0:
				break
			move[:] = [int(input('Index invalid. Select next move: '))]

class human2(connect4Player):

	def play(self, env, move):
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, RED, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, YELLOW, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move[:] = [col]
					done = True

class randomAI(connect4Player):

	def play(self, env, move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move[:] = [random.choice(indices)]

class stupidAI(connect4Player):

	def play(self, env, move):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move[:] = [3]
		elif 2 in indices:
			move[:] = [2]
		elif 1 in indices:
			move[:] = [1]
		elif 5 in indices:
			move[:] = [5]
		elif 6 in indices:
			move[:] = [6]
		else:
			move[:] = [0]

class minimaxAI(connect4Player):
	TURN = 0
	PLAYON = -1
	WHITE_BOARD = np.array([[0,0,0,0,0,0,0],
						   [0,1,2,5,2,1,0],
						   [1,3,10,11,10,3,1],
						   [5,16,20,21,20,16,5],
						   [1,10,15,21,15,10,1],
						   [1,7,20,25,20,7,1]])
	BLACK_BOARD = np.array([[0,0,0,0,0,0,0],
						   [0,1,0,2,0,1,0],
						   [1,3,13,15,13,3,1],
						   [1,6,12,17,12,6,1],
						   [5,19,23,24,23,19,5],
						   [1,4,20,25,20,4,1]])

	def start_MAX(self, last, topPosition, board, total, depth, eval, white_groups, black_groups):
		over, g = self.gameOver(topPosition, board, last[-1], 2, 1)
		if over:
			return None, -math.inf
		if total == 42:
			return None, 0
		if depth == 0:
			return None, eval + 3 * (white_groups[2] - (black_groups + g)[2]) + 7 * (white_groups[3] - (black_groups + g)[3])
			# return None, eval

		v = -math.inf
		c = -1
		for col in self.get_valid(topPosition):
			myboard = deepcopy(board)
			myTopPosition = deepcopy(topPosition)
			mylast = deepcopy(last)
			mylast.append(col)
			mygroups = deepcopy(black_groups)
			r = myTopPosition[col]
			self.simulate_move(myTopPosition, myboard, col, 1)
			temp = self.start_MIN(mylast, myTopPosition, myboard, total + 1, depth - 1, eval + self.WHITE_BOARD[r, col], white_groups, mygroups + g)[1]
			if temp == math.inf: return col, temp
			if temp > v:
				v = temp
				c = col
		return c, v

	def start_MIN(self, last, topPosition, board, total, depth, eval, white_groups, black_groups):
		over, g = self.gameOver(topPosition, board, last[-1], 1, 2)
		if over:
			return None, math.inf
		if depth == 0:
			return None, eval + 3 * ((white_groups + g)[2] - black_groups[2]) + 7 * ((white_groups + g)[3] - black_groups[3])
			# return None, eval

		v = math.inf
		c = -1
		for col in self.get_valid(topPosition):
			myboard = deepcopy(board)
			myTopPosition = deepcopy(topPosition)
			mylast = deepcopy(last)
			mylast.append(col)
			mygroups = deepcopy(white_groups)
			r = myTopPosition[col]
			self.simulate_move(myTopPosition, myboard, col, 2)
			temp = self.start_MAX(mylast, myTopPosition, myboard, total + 1, depth - 1, eval - self.BLACK_BOARD[r, col], mygroups + g, black_groups)[1]
			if temp == -math.inf: return col, temp
			if temp < v:
				v = temp
				c = col
		return c, v

	def start_play(self, env, start_depth, eval):
		l = len(env.history[0]) + len(env.history[1])
		groups = np.array([0,0,0,0])
		if env.turnPlayer.position == 1:
			c, e = self.start_MAX([env.history[1][-1]], env.topPosition, env.board, l, start_depth, eval, groups, groups)
			if e == -math.inf:
				c = -1
			return c, e
		else:
			c, e = self.start_MIN([env.history[0][-1]], env.topPosition, env.board, l, start_depth, eval, groups, groups)
			if e == math.inf:
				c = -1
			return c, e

	def get_starting_eval(self, env):
		eval = 0
		for i in range(1, 6):
			for j in range(7):
				if env.board[i, j] == 1:
					eval += self.WHITE_BOARD[i, j]
				elif env.board[i, j] == 2:
					eval -= self.BLACK_BOARD[i, j]
		return eval

	def play(self, env, move):
		move[:] = [3]
		TURN = len(env.history[0])
		start_depth = 4
		closed = list(env.topPosition >= 0).count(False)
		if closed >= 2:
			start_depth += 2 ** (closed - 2)
		myenv = env.getEnv()
		myenv.visualize = False
		if TURN == 0: return
		col, eval = self.start_play(myenv, start_depth, self.get_starting_eval(env))
		if col == -1:
			start_depth -= 1
			while col == -1 and start_depth > 0:
				col, eval = self.start_play(myenv, start_depth, self.get_starting_eval(env))
				start_depth -= 1
		self.PLAYON = col
		move[:] = [col]
		print(TURN, "done", eval, start_depth)
		col, eval = self.start_play(myenv, start_depth + 1, self.get_starting_eval(env))
		if col == -1:
			col = self.PLAYON
		move[:] = [col]
		print(TURN, "done", eval, start_depth + 1)

	def get_valid(self, topPosition):
		possible = topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		return indices

	def simulate_move(self, topPosition, board, move, player):
		board[topPosition[move]][move] = player
		topPosition[move] -= 1

	def gameOver(self, topPosition, board, j, player, opp):
		i = topPosition[j] + 1 # i is row of where i just put piece, j is the col
		groups = np.array([0,0,0,0])
		# horizontal
		minColIndex = max(j - 3, 0)
		maxColIndex = min(j + 3, 6)
		win_count = 0
		window_size = 0
		window_count = 0
		s = minColIndex
		while s < maxColIndex + 1:
			if board[i, s] == opp:
				win_count = 0
				window_size = -1
				window_count = 0
				if s > 2 or s > j:
					break
			else:
				if board[i, s] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[i, s - 4]
					groups[window_count] += 1
			s += 1
			window_size += 1

		# vertical
		win_count = 0
		if i <= 2:
			for s in range(4):
				if board[i + s, j] == player:
					win_count += 1
				else:
					break
			if win_count == 4:
				return True, None

		#tlbr diag
		row = i
		col = j
		win_count = 0
		window_size = 0
		window_count = 0
		run_count = 0
		while row > -1 and col > -1 and run_count < 4:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row + 4, col + 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row -= 1
			col -= 1

		run_count = 0
		row = i + 1
		col = j + 1
		while row < 6 and col < 7 and run_count < 3:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row - 4, col - 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row += 1
			col += 1

		#bltr diag
		row = i
		col = j
		win_count = 0
		window_size = 0
		window_count = 0
		run_count = 0
		while row < 6 and col > -1 and run_count < 4:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row - 4, col + 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row += 1
			col -= 1

		run_count = 0
		row = i - 1
		col = j + 1
		while row > -1 and col < 7 and run_count < 3:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row + 4, col - 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row -= 1
			col += 1
		return False, groups

class alphaBetaAI(connect4Player):
	TURN = 0
	PLAYON = -1
	WHITE_BOARD = np.array([[0,0,0,0,0,0,0],
						   [0,1,2,5,2,1,0],
						   [1,3,10,11,10,3,1],
						   [5,16,20,21,20,16,5],
						   [1,10,15,21,15,10,1],
						   [1,7,20,25,20,7,1]])
	BLACK_BOARD = np.array([[0,0,0,0,0,0,0],
						   [0,1,0,2,0,1,0],
						   [1,3,13,15,13,3,1],
						   [1,6,12,17,12,6,1],
						   [5,19,23,24,23,19,5],
						   [1,4,20,25,20,4,1]])
	POS = [None, WHITE_BOARD, BLACK_BOARD]

	def start_MAX(self, last, topPosition, board, total, depth, eval, white_groups, black_groups, alpha, beta):
		over, g = self.gameOver(topPosition, board, last[-1], 2, 1)
		if over:
			return None, -math.inf
		if total == 42:
			return None, 0
		if depth == 0:
			return None, eval + 3 * (white_groups[2] - (black_groups + g)[2]) + 7 * (white_groups[3] - (black_groups + g)[3])
			# return None, eval

		v = -math.inf
		c = -1
		for col in self.get_valid(topPosition, 1):
			myboard = deepcopy(board)
			myTopPosition = deepcopy(topPosition)
			mylast = deepcopy(last)
			mylast.append(col)
			mygroups = deepcopy(black_groups)
			r = myTopPosition[col]
			self.simulate_move(myTopPosition, myboard, col, 1)
			temp = self.start_MIN(mylast, myTopPosition, myboard, total + 1, depth - 1, eval + self.WHITE_BOARD[r, col], white_groups, mygroups + g, alpha, beta)[1]
			if temp > v:
				v = temp
				c = col
			if v >= beta: return c, v
			alpha = max(alpha, v)
		return c, v

	def start_MIN(self, last, topPosition, board, total, depth, eval, white_groups, black_groups, alpha, beta):
		over, g = self.gameOver(topPosition, board, last[-1], 1, 2)
		if over:
			return None, math.inf
		if depth == 0:
			return None, eval + 3 * ((white_groups + g)[2] - black_groups[2]) + 7 * ((white_groups + g)[3] - black_groups[3])
			# return None, eval

		v = math.inf
		c = -1
		for col in self.get_valid(topPosition, 2):
			myboard = deepcopy(board)
			myTopPosition = deepcopy(topPosition)
			mylast = deepcopy(last)
			mylast.append(col)
			mygroups = deepcopy(white_groups)
			r = myTopPosition[col]
			self.simulate_move(myTopPosition, myboard, col, 2)
			temp = self.start_MAX(mylast, myTopPosition, myboard, total + 1, depth - 1, eval - self.BLACK_BOARD[r, col], mygroups + g, black_groups, alpha, beta)[1]
			if temp < v:
				v = temp
				c = col
			if v <= alpha: return c, v
			beta = min(beta, v)
		return c, v

	def start_play(self, env, start_depth, eval):
		l = len(env.history[0]) + len(env.history[1])
		groups = np.array([0,0,0,0])
		a = -math.inf
		b = math.inf
		if env.turnPlayer.position == 1:
			c, e = self.start_MAX([env.history[1][-1]], env.topPosition, env.board, l, start_depth, eval, groups, groups, a, b)
			if e == -math.inf:
				c = -1
			return c, e
		else:
			c, e = self.start_MIN([env.history[0][-1]], env.topPosition, env.board, l, start_depth, eval, groups, groups, a, b)
			if e == math.inf:
				c = -1
			return c, e

	def get_starting_eval(self, env):
		eval = 0
		for i in range(1, 6):
			for j in range(7):
				if env.board[i, j] == 1:
					eval += self.WHITE_BOARD[i, j]
				elif env.board[i, j] == 2:
					eval -= self.BLACK_BOARD[i, j]
		return eval

	def play(self, env, move):
		#to do: finish running minimax, try out alphabeta on csif, potentially change order of max/min for gameover function to use in successor
		move[:] = [3]
		TURN = len(env.history[0])
		start_depth = 4
		closed = list(env.topPosition >= 0).count(False)
		if closed >= 2:
			start_depth += 2 ** (closed - 2)
		myenv = env.getEnv()
		myenv.visualize = False
		if TURN == 0: return
		col, eval = self.start_play(myenv, start_depth, self.get_starting_eval(env))
		if col == -1:
			start_depth -= 1
			while col == -1 and start_depth > 0:
				col, eval = self.start_play(myenv, start_depth, self.get_starting_eval(env))
				start_depth -= 1
		self.PLAYON = col
		move[:] = [col]
		print(TURN, "done", eval, start_depth)
		i = 1
		while True:
			col, eval = self.start_play(myenv, start_depth + i, self.get_starting_eval(env))
			if col == -1:
				col = self.PLAYON
			move[:] = [col]
			i += 1

	def get_valid(self, topPosition, player):
		possible = topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		for i in range(1, len(indices)):
			col = indices[i]
			col_compare = self.POS[player][topPosition[col]][col]
			j = i - 1
			j_compare = self.POS[player][topPosition[indices[j]]][indices[j]]
			while j_compare < col_compare:
				indices[j + 1] = indices[j]
				j = j - 1
				if j < 0: break
				j_compare = self.POS[player][topPosition[indices[j]]][indices[j]]
			indices[j + 1] = col
		return indices

	def simulate_move(self, topPosition, board, move, player):
		board[topPosition[move]][move] = player
		topPosition[move] -= 1

	def gameOver(self, topPosition, board, j, player, opp):
		i = topPosition[j] + 1 # i is row of where i just put piece, j is the col
		groups = np.array([0,0,0,0])
		# horizontal
		minColIndex = max(j - 3, 0)
		maxColIndex = min(j + 3, 6)
		win_count = 0
		window_size = 0
		window_count = 0
		s = minColIndex
		while s < maxColIndex + 1:
			if board[i, s] == opp:
				win_count = 0
				window_size = -1
				window_count = 0
				if s > 2 or s > j:
					break
			else:
				if board[i, s] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[i, s - 4]
					groups[window_count] += 1
			s += 1
			window_size += 1

		# vertical
		win_count = 0
		if i <= 2:
			for s in range(4):
				if board[i + s, j] == player:
					win_count += 1
				else:
					break
			if win_count == 4:
				return True, None

		#tlbr diag
		row = i
		col = j
		win_count = 0
		window_size = 0
		window_count = 0
		run_count = 0
		while row > -1 and col > -1 and run_count < 4:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row + 4, col + 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row -= 1
			col -= 1

		run_count = 0
		row = i + 1
		col = j + 1
		while row < 6 and col < 7 and run_count < 3:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row - 4, col - 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row += 1
			col += 1

		#bltr diag
		row = i
		col = j
		win_count = 0
		window_size = 0
		window_count = 0
		run_count = 0
		while row < 6 and col > -1 and run_count < 4:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row - 4, col + 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row += 1
			col -= 1

		run_count = 0
		row = i - 1
		col = j + 1
		while row > -1 and col < 7 and run_count < 3:
			if board[row, col] == opp:
				break
			else:
				if board[row, col] == player:
					win_count += 1
					if win_count == 4:
						return True, None
					window_count += 1
				if window_size == 3:
					groups[window_count] += 1
				if window_size >= 4:
					window_count -= board[row + 4, col - 4]
					groups[window_count] += 1
			window_size += 1
			run_count += 1
			row -= 1
			col += 1
		return False, groups

SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)




