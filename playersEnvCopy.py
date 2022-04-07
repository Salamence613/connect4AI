import random
import sys
import time

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
	# to do:
	# 1. evaluation function using previous function and implementing double threats and 3 horiz
	# 2. create lowest odd/even threat and then just play defense (# of 2/3 in a rows) with increased depth
	TURN = 0
	WHITE_BOARD = np.array([[0,0,0,0,0,0,0],
						   [0,0,0,0,0,0,0],
						   [1,2,10,11,10,2,1],
						   [5,16,20,21,20,16,5],
						   [1,6,15,21,15,6,1],
						   [1,7,20,25,20,7,1]])
	BLACK_BOARD = np.array([[0,0,0,0,0,0,0],
						   [0,0,0,0,0,0,0],
						   [1,3,13,15,13,3,1],
						   [1,2,9,10,9,2,1],
						   [5,19,23,24,23,19,5],
						   [1,4,20,25,20,4,1]])

	# def evaluation(self, env): return 0
	#
	# def MAX(self, env, depth):
	# 	if self.gameOver(env, env.history[1][-1], 2) : return None, -math.inf
	# 	if len(env.history[0]) + len(env.history[1]) == env.shape[0]*env.shape[1]: return None, 0
	# 	if depth == 0: return None, self.evaluation(env)
	#
	# 	v = -math.inf
	# 	c = -1
	# 	for col in self.get_valid(env):
	# 		myenv = env.getEnv()
	# 		self.simulate_move(myenv, col, 1)
	# 		temp = self.MIN(myenv, depth - 1)[1]
	# 		if temp > v:
	# 			v = temp
	# 			c = col
	# 	return c, v
	#
	# def MIN(self, env, depth):
	# 	if self.gameOver(env, env.history[0][-1], 1): return None, math.inf
	# 	if depth == 0: return None, self.evaluation(env)
	#
	# 	v = math.inf
	# 	c = -1
	# 	for col in self.get_valid(env):
	# 		myenv = env.getEnv()
	# 		self.simulate_move(myenv, col, 2)
	# 		temp = self.MAX(myenv, depth - 1)[1]
	# 		if temp < v:
	# 			v = temp
	# 			c = col
	# 	return c, v

	def start_MAX(self, env, depth, eval):
		# if self.gameOver(env, env.history[1][-1], 2) : return None, -math.inf
		# print(env.history[1][-1], "MIN")
		if env.gameOver(env.history[1][-1], 2) : return None, -math.inf
		if len(env.history[0]) + len(env.history[1]) == env.shape[0]*env.shape[1]: return None, 0
		if depth == 0: return None, eval

		v = -math.inf
		c = -1
		for col in self.get_valid(env):
			myenv = env.getEnv()
			r = env.topPosition[col]
			self.simulate_move(myenv, col, 1)
			temp = self.start_MIN(myenv, depth - 1, eval + self.WHITE_BOARD[r, col])[1]
			if temp == math.inf: return col, temp
			if temp > v:
				v = temp
				c = col
		return c, v

	def start_MIN(self, env, depth, eval):
		# if self.gameOver(env, env.history[0][-1], 1): return None, math.inf
		# print(env.history[0][-1], "MAX")
		if env.gameOver(env.history[0][-1], 1): return None, math.inf
		if depth == 0: return None, eval

		v = math.inf
		c = -1
		for col in self.get_valid(env):
			myenv = env.getEnv()
			r = env.topPosition[col]
			self.simulate_move(myenv, col, 2)
			temp = self.start_MAX(myenv, depth - 1, eval - self.BLACK_BOARD[r, col])[1]
			if temp == -math.inf: return col, temp
			if temp < v:
				v = temp
				c = col
		return c, v

	def start_play(self, env, start_depth, eval):
		if env.turnPlayer.position == 1:
			return self.start_MAX(env, start_depth, eval)
		else:
			return self.start_MIN(env, start_depth, eval)

	def get_starting_eval(self, env):
		eval = 0
		for i in range(2, 6):
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
		depth = 1
		myenv = env.getEnv()
		myenv.visualize = False
		if TURN == 0: return
		if TURN <= 20:
			col, eval = self.start_play(myenv, start_depth, self.get_starting_eval(env))
		# else:
		# 	if env.turnPlayer.position == 1:
		# 		col, eval = self.MAX(env, depth)
		# 	else:
		# 		col, eval = self.MIN(env, depth)
		move[:] = [col]
		print("done")

	def get_valid(self, env):
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		return indices

	def simulate_move(self, env, move, player):
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[player - 1].append(move)

	def gameOver(self, env, j, player):
		i = env.topPosition[j] + 1
		minRowIndex = max(j - 3, 0)
		maxRowIndex = min(j + 3, env.shape[1] - 1)
		maxColumnIndex = max(i - 3, 0)
		minColumnIndex = min(i + 3, env.shape[0] - 1)
		count = 0
		for s in range(minRowIndex, maxRowIndex + 1):
			if env.board[i, s] == player:
				count += 1
			else:
				count = 0
			if count == 4:
				return True
		count = 0
		for s in range(maxColumnIndex, minColumnIndex + 1):
			if env.board[s, j] == player:
				count += 1
			else:
				count = 0
			if count == 4:
				return True
		row = i
		col = j
		count = 0
		while row > -1 and col > -1 and env.board[row][col] == player:
			count += 1
			row -= 1
			col -= 1
		down_count = count
		row = i + 1
		col = j + 1
		while row < env.shape[0] and col < env.shape[1] and env.board[row][col] == player:
			count += 1
			row += 1
			col += 1
		if count >= 4:
			return True
		row = i
		col = j
		count = 0
		while row < env.shape[0] and col > -1 and env.board[row][col] == player:
			count += 1
			row += 1
			col -= 1
		down_count = count
		row = i - 1
		col = j + 1
		while row > -1 and col < env.shape[1] and env.board[row][col] == player:
			count += 1
			row -= 1
			col += 1
		if count >= 4:
			return True
		return False


class alphaBetaAI(connect4Player):

	def play(self, env, move):
		pass


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




