import chess
import pygame
import torch
from DeepNN import DeepNN
from Partie import Partie
from MCTS import mcts_search
from Train import Train

class Main:
    def __init__(self):
        self.model = DeepNN()  # Initialiser le modèle de l'IA
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)  # Optimiseur pour entraîner le modèle
        self.trainer = Train(self.model)
        self.self_play_data = []  # Liste pour stocker les données d'auto-apprentissage

    def jouer_une_partie(self):
        """Fait jouer une partie complète entre l'IA contre elle-même."""
        partie = Partie(self.model, display=True)  # Créer une partie avec affichage
        partie.jouer_partie()  # L'IA joue contre elle-même
        return partie.get_result()  # Récupérer le résultat de la partie

    def self_play(self, num_parties):
        """Jouer plusieurs parties en auto-apprentissage pour générer des données d'entraînement."""
        for i in range(num_parties):
            print(f"Début de la partie {i + 1}")
            self.trainer.play_game(self.model)  # Jouer une partie en auto-apprentissage
            if len(self.self_play_data) > 500:  # Si on a suffisamment de données
                self.trainer.train(self.model, self.self_play_data, self.optimizer)  # Entraîner le modèle avec les données

    def main_loop(self):
        """Boucle principale pour faire jouer l'IA et afficher les résultats."""
        while True:
            # Jouer une partie contre elle-même et afficher le résultat
            resultat = self.jouer_une_partie()
            print(f"Résultat final : {'Blancs gagnent' if resultat == 1 else 'Noirs gagnent' if resultat == -1 else 'Égalité'}")
            
            # Ajouter de l'auto-apprentissage pour améliorer l'IA
            self.self_play(5)  # Par exemple, jouer 5 parties en auto-apprentissage

if __name__ == "__main__":
    main_game = Main()  # Initialiser le jeu
    main_game.main_loop()  # Lancer la boucle principale du jeu
