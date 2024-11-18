import chess
import numpy as np

#convert chess position to tensor
def board_to_tensor(board):
    tensor = np.zeros((8, 8, 101), dtype=np.float32) 
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
        chess.QUEEN: 4, chess.KING: 5
    }
    #fill tensor with piece values
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        #if there is a piece on the square fill tensor with piece value
        if piece:
            plane = piece_map[piece.piece_type] + (0 if piece.color == chess.WHITE else 6) #0-5 for white pieces, 6-11 for black pieces
            row, col = divmod(square, 8)
            tensor[row, col, plane] = 1 
    return tensor
