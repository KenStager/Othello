from .board import Board

class Game:
    def __init__(self, board_size=8):
        self.n = board_size

    def new_board(self):
        return Board(self.n)
