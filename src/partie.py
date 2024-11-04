import pygame
import chess
import chess.svg

# Initialiser Pygame et le plateau d'échecs
pygame.init()
screen = pygame.display.set_mode((640, 640))
pygame.display.set_caption("Jeu d'Échecs - python-chess + Pygame")

square_width = 80
square_height = 80
# Charger les images des pièces
piece_names = ["pion_noir", "tour_noir", "cheval_noir", "fou_noir", "reine_noir", "roi_noir", 
                "pion_blanc", "tour_blanc", "cheval_blanc", "fou_blanc", "reine_blanc", "roi_blanc"]
piece_images = {
    "P": pygame.transform.scale(pygame.image.load("../images/pion_blanc.png").convert_alpha(), (square_width, square_height)),
    "R": pygame.transform.scale(pygame.image.load("../images/tour_blanc.png").convert_alpha(), (square_width, square_height)),
    "N": pygame.transform.scale(pygame.image.load("../images/cheval_blanc.png").convert_alpha(), (square_width, square_height)),
    "B": pygame.transform.scale(pygame.image.load("../images/fou_blanc.png").convert_alpha(), (square_width, square_height)),
    "Q": pygame.transform.scale(pygame.image.load("../images/reine_blanc.png").convert_alpha(), (square_width, square_height)),
    "K": pygame.transform.scale(pygame.image.load("../images/roi_blanc.png").convert_alpha(), (square_width, square_height)),
    "p": pygame.transform.scale(pygame.image.load("../images/pion_noir.png").convert_alpha(), (square_width, square_height)),
    "r": pygame.transform.scale(pygame.image.load("../images/tour_noir.png").convert_alpha(), (square_width, square_height)),
    "n": pygame.transform.scale(pygame.image.load("../images/cheval_noir.png").convert_alpha(), (square_width, square_height)),
    "b": pygame.transform.scale(pygame.image.load("../images/fou_noir.png").convert_alpha(), (square_width, square_height)),
    "q": pygame.transform.scale(pygame.image.load("../images/reine_noir.png").convert_alpha(), (square_width, square_height)),
    "k": pygame.transform.scale(pygame.image.load("../images/roi_noir.png").convert_alpha(), (square_width, square_height)),
}

for piece in piece_names:
    piece_images[piece] = pygame.image.load(f"../images/{piece}.png").convert_alpha()


# Créer le plateau
board = chess.Board()

# Fonction pour dessiner le plateau
def draw_board():
    colors = [pygame.Color(235, 235, 208), pygame.Color(119, 149, 86)]
    for row in range(8):
        for col in range(8):
            color = colors[(row + col) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(col * 80, row * 80, 80, 80))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x = (square % 8) * 80
            y = (7 - (square // 8)) * 80
            screen.blit(piece_images[piece.symbol()], (x, y))

selected_square = None

# Boucle de jeu
running = True
while running:
    draw_board()
    pygame.display.flip()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            col, row = x // 80, 7 - (y // 80)
            square = chess.square(col, row)

            if selected_square is None:
                # Sélectionner la pièce à déplacer
                if board.piece_at(square):
                    selected_square = square
            else:
                # Essayer de déplacer la pièce
                move = chess.Move(selected_square, square)
                if move in board.legal_moves:
                    board.push(move)
                selected_square = None

    # Vérifier si la partie est terminée
    if board.is_game_over():
        print("Fin de la partie :", board.result())
        running = False

pygame.quit()
