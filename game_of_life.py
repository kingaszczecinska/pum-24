import random 
import time
from IPython.display import clear_output
import os 
import argparse 

args = parser.parse_args()
args.size   # this will contain the size of the board
args.prob   # this will contain the probability of a cell being alive
args.steps  # this will contain the number of steps to run the simulation for

parser = argparse.ArgumentParser()
parser.add_argument("--size", "-s", type=int, default=10, help="Size of the board")
parser.add_argument("--prob", "-p", type=float, default=0.2, help="Probability of a cell being alive")
parser.add_argument("--steps", "-n", type=int, default=20, help="Number of steps to run the simulation for")


def get_empty_board(n):# return n x n table of dead cells (a list of lists)
    empty_board = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        empty_board.append(row)       
    return empty_board
def print_board(grid): # print the table
    for row in grid:
        nice_row = ' '.join(str(cell) for cell in row)
        nice_row = nice_row.replace('0', '.')
        nice_row = nice_row.replace('1', 'X')
        print(nice_row)

def get_random_board(n, p=0.2): # return n x n table where each cell is alive with probability 0.2
    empty = get_empty_board(n)
    for row in range(n):
        for element in range(n):
            random_number = random.random()
            if random_number > args.prob:
                empty[row][element]=0
            else: 
                empty[row][element]=1
    return empty
    
def add_glider(board): # return a board with a glider
    board[0][0]=1
    board[1][1]=1
    board[1][2]=1
    board[2][0]=1
    board[2][1]=1
    return board
    
def count_live_neighbors(board, x, y): # return the number of live neighbors of cell x, y
    sum=0
    for i in [x-1,x,x+1]:
        for j in [y-1,y,y+1]:
            if i< len(board) and j<len(board):
                sum += board[i][j]
    if board[x][y]==1:
        sum-=1
    return sum
    
def step(board): # return board at the next timestep
    n=len(board)
    new_board = get_empty_board(n)
    
    for i in range(n):
        for j in range(n):
            neighbours = count_live_neighbors(board, i, j)
            if board[i][j]==1:
                if neighbours<2:
                    new_board[i][j]=0
                elif neighbours in [2,3]:
                    new_board[i][j]=1
                else:
                    new_board[i][j]=0
            else:
                if neighbours==3:
                    new_board[i][j]=1
    return new_board

board = get_random_board(10)
#board = get_empty_board(10)
board = add_glider(board) # add a glider to the board

for _ in range(args.steps):    # run for 20 steps
    os.system('cls')    # clear the output
    print_board(board)          # print the board
    time.sleep(2)               # wait for half a second
    new_board = step(board)     # generate the next step
    board = new_board           # update the board

