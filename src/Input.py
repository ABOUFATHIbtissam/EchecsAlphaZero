import numpy as np
import chess
import torch


piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
        chess.QUEEN: 4, chess.KING: 5
    }

class chessBoardAlphaZero :
    
    def __init__(self, board):
        if board == None:
            self.white = None
            self.black = None
            self.turn = None
            self.white_castling = None
            self.black_castling = None
        else:
            self.white = [self.board_to_tab(board,1)] * 8
            self.black = [self.board_to_tab(board,0)] * 8
            self.turn = np.ones((8,8,1), dtype=np.float32)
            self.white_castling = np.zeros((8,8,2), dtype=np.float32)
            self.black_castling = np.zeros((8,8,2), dtype=np.float32)
    
    def to_tensor(self):
        tensor = None
        tensor_white = self.prepare_list(self.white)
        tensor_black = self.prepare_list(self.black)
        
        tensor=np.concatenate((tensor_white, tensor_black), axis=-1)  
        tensor=np.concatenate((tensor, self.turn), axis=-1)
        tensor=np.concatenate((tensor, self.white_castling), axis=-1)
        tensor=np.concatenate((tensor, self.black_castling), axis=-1)
        
        return torch.tensor(tensor)   
        
            
    
    def board_to_tab(self, board, turn):
        list = np.zeros((8,8,6), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                piece = board.piece_at(8*i + j)
                if piece:
                    if piece.color == turn:
                        list[i, j, piece.piece_type - 1] = 1
        return list
    
    def prepare_list(self, list):
        tlist = np.zeros((8,8,48), dtype=np.float32)
        for i in range(8):
            tlist[:,:,i*6:6*(i+1)] = list[i]
        return tlist
        
    def update(self, board):
        turn = int(self.turn[0,0,0])
        
        if turn == 1:
            self.white.insert(0,self.board_to_tab(board,turn))
            self.white.pop()
            self.turn = np.zeros((8,8,1), dtype=np.float32)
            self.white_castling[:,:,0] = board.has_kingside_castling_rights(turn)
            self.white_castling[:,:,1] = board.has_queenside_castling_rights(turn)
            
        else:
            self.black.insert(0,self.board_to_tab(board,turn))
            self.black.pop()
            self.turn = np.ones((8,8,1), dtype=np.float32)
            self.black_castling[:,:,0] = board.has_kingside_castling_rights(turn)
            self.black_castling[:,:,1] = board.has_queenside_castling_rights(turn)
            
    def copy(self):
        new_self = chessBoardAlphaZero(None)
        new_self.white = self.white
        new_self.black = self.black
        new_self.turn = self.turn
        new_self.white_castling = self.white_castling
        new_self.black_castling = self.black_castling
        return new_self