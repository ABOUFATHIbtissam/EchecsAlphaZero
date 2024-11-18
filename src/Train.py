import chess
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import json
from MCTS import mcts_search
from TransformTensor import board_to_tensor

#Self-Play qui joue des parties contre lui-même pour générer des données d'entraînement

class Train:
    def __init__(self, model, learning_rate=0.001, max_memory=10000, batch_size=64, epochs=100):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.self_play_data = deque(maxlen=max_memory)
        self.batch_size = batch_size
        self.epochs = epochs

    def play_game(self, model):
        """Fait jouer une partie complète à l'IA contre elle-même pour collecter des données."""

        board = chess.Board()
        game_data = []
        simulations_data = []

        while not board.is_game_over():
            # Exécute MCTS pour obtenir le meilleur coup et la distribution de probabilité des coups
            move, policy_distribution, _, simulations = mcts_search(board, model)
            print("game_data ", game_data)
            # Enregistrez les simulations ici
            simulations_data.append(simulations)

            # Stocker la position, la distribution de MCTS et le résultat temporaire (0)
            # `board_to_tensor` convertit la position du plateau en tenseur pour l'entraînement
            game_data.append((board_to_tensor(board), policy_distribution, 0)) #etat jeu, distribution des proba des coups, resultat de la partie
            
            #Jouer le coup sur le plateau
            board.push(move)
        
        # Mettre à jour les résultats finaux de la partie
        # Résultat de la partie (1 = victoire blanche, -1 = victoire noire, 0 = égalité)
        result = 1 if board.result() == "1-0" else -1 if board.result() == "0-1" else 0
        for i in range(len(game_data)):
            state, policy, _ = game_data[i]
            game_data[i] = (state, policy, result)
        
        # Ajouter la partie dans la mémoire de self-play
        self.self_play_data.extend(game_data)
        # Enregistrer les simulations de cette partie dans un fichier ou afficher
        self.save_simulations(simulations_data)
        print("Partie terminée et données ajoutées à la mémoire.")


    def train(self):
        """Entraîne le modèle avec les données générées lors des parties autonomes."""
        self.model.train()
        
        for epoch in range(self.epochs):  
            if len(self.self_play_data) < self.batch_size:
                continue  # Vérification pour s'assurer qu'on a suffisamment de données
            
            # Tirer un batch de données aléatoirement
            batch = random.sample(self.self_play_data, self.batch_size)
            states, policy_targets, value_targets = zip(*batch) # Diviser les données en trois listes différentes pour les états, les cibles de politique et les cibles de valeur
            
            # Préparer les tenseurs
            states = torch.stensor(states, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            
            # Prédictions du modèle
            predicted_policy, predicted_value = self.model(states)
            
            # Calcul des pertes
            policy_loss = -torch.sum(policy_targets * F.log_softmax(predicted_policy, dim=1)) / self.batch_size # Cross-entropy loss qui mesure la différence entre les distributions de probabilité
            value_loss = F.mse_loss(predicted_value.view(-1), value_targets) #fonction de perte de l'erreur quadratique moyenne pour la valeur
            loss = policy_loss + value_loss 
            
            # Backpropagation et mise à jour des poids
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Afficher la perte pour suivre l'entraînement
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")

    def run_training_loop(self, num_games=1000, eval_interval=100):
        """Exécute le cycle complet de self-play, d'entraînement et d'évaluation."""
        best_model = self.model

        for game in range(num_games):
            print(f"--- Partie {game + 1} ---")
            self.play_game(best_model)  # l'auto-jouer pour générer des données de self-play
            self.train()  # Entraîner avec les données de self-play

            # Évaluation périodique
            if (game + 1) % eval_interval == 0:
                print("Évaluation du modèle...")
                new_model = self.model
                if self.evaluate_model(new_model, best_model):
                    print("Nouveau modèle adopté.")
                    best_model = new_model  #Remplace l'ancien modèle
                    self.save_model()
                else:
                    print("Ancien modèle conservé.")

            print(f"Fin de la partie {game + 1}")

    def save_simulations(self, simulations_data):
        """Sauvegarde les simulations dans un fichier JSON."""
        with open("simulations.json", "a") as file:
            json.dump(simulations_data, file)
            file.write("\n")
        print(f"Simulations enregistrées pour cette partie.")

    def save_model(self, file_path="best_model.pth"):
        """Sauvegarde le modèle dans un fichier."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Modèle sauvegardé dans {file_path}.")

    def load_model(self, file_path="best_model.pth"):
        """Charge un modèle à partir d'un fichier."""
        self.model.load_state_dict(torch.load(file_path))
        print(f"Modèle chargé depuis {file_path}.")

    def evaluate_model(self, new_model, old_model, num_games=10):
        """Évalue un nouveau modèle contre l'ancien sur un certain nombre de parties."""
        wins, losses, draws = 0, 0, 0

        for _ in range(num_games):
            board = chess.Board()
            current_player = new_model

            while not board.is_game_over():
                move, _, _, _ = mcts_search(board, current_player)
                board.push(move)

                # Alterner les modèles
                current_player = old_model if current_player == new_model else new_model

            result = 1 if board.result() == "1-0" else -1 if board.result() == "0-1" else 0
            if result == 1:
                wins += 1
            elif result == -1:
                losses += 1
            else:
                draws += 1

        win_rate = wins / num_games
        print(f"Win rate: {win_rate:.2%} | Wins: {wins}, Losses: {losses}, Draws: {draws}")
        return win_rate >= 0.55  # Remplace l'ancien modèle si taux de victoire >= 55 %

