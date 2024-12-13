import chess
import pygame
from MCTS import mcts_search
from Input import chessBoardAlphaZero

class Partie:
    def __init__(self, model, display=False):
        """
        Initialise une nouvelle partie d'échecs.
        :param model: Le modèle de réseau neuronal (DeepNN) utilisé pour les décisions de jeu.
        :param display: Activer ou désactiver l'affichage de la partie (optionnel, utile pour visualiser les jeux).
        """
        self.board = chess.Board()  # Créer un plateau d'échecs vide
        self.boardToTensor = chessBoardAlphaZero(self.board)
        self.model = model
        self.display = display
        if display:
            pygame.init()
            self.screen = pygame.display.set_mode((640, 640))
            pygame.display.set_caption("Partie d'Échecs - IA vs IA")
            self.square_size = 80

    def draw_board(self):
        """Affiche le plateau et les pièces dans Pygame."""
        colors = [pygame.Color(235, 235, 208), pygame.Color(119, 149, 86)]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(self.screen, color, pygame.Rect(col * self.square_size, row * self.square_size, self.square_size, self.square_size))
        
        # Placer les images des pièces
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x = (square % 8) * self.square_size
                y = (7 - (square // 8)) * self.square_size
                piece_image = self.get_piece_image(piece.symbol())  # Fonction pour charger les images des pièces
                self.screen.blit(piece_image, (x, y))
        pygame.display.flip()

    def get_piece_image(self, piece_symbol):
        """Charge l'image de la pièce correspondant au symbole (P = pion blanc, r = tour noir, etc.)."""
        piece_images = {
            "P": "../images/pion_blanc.png",
            "R": "../images/tour_blanc.png",
            "N": "../images/cheval_blanc.png",
            "B": "../images/fou_blanc.png",
            "Q": "../images/reine_blanc.png",
            "K": "../images/roi_blanc.png",
            "p": "../images/pion_noir.png",
            "r": "../images/tour_noir.png",
            "n": "../images/cheval_noir.png",
            "b": "../images/fou_noir.png",
            "q": "../images/reine_noir.png",
            "k": "../images/roi_noir.png"
        }
        image_path = piece_images[piece_symbol]
        return pygame.transform.scale(pygame.image.load(image_path).convert_alpha(), (self.square_size, self.square_size))

    def jouer_coup(self):
        """Exécute un coup via MCTS et met à jour le plateau."""
        if self.board.is_game_over():
            return None

        move, _, _, _, _ = mcts_search(self.board, self.boardToTensor, self.model)  # Obtenir le meilleur coup avec MCTS
        self.board.push(move)  # Appliquer le coup

        if self.display:
            self.draw_board()  # Mettre à jour l'affichage

        return move

    def jouer_partie(self):
        """Lance la partie complète jusqu'à la fin."""
        if self.display:
            self.draw_board()
        
        while not self.board.is_game_over():
            self.jouer_coup()
        
        # Fin de la partie
        print("Résultat de la partie :", self.board.result())
        if self.display:
            pygame.quit()

    def get_result(self):
        """Renvoie le résultat de la partie : 1 si les Blancs gagnent, -1 si les Noirs gagnent, 0 si égalité."""
        if self.board.result() == "1-0":
            return 1
        elif self.board.result() == "0-1":
            return -1
        else:
            return 0
