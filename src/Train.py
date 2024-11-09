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
        board = chess.Board()
        game_data = []
        simulations_data = []

        while not board.is_game_over():
            # Exécute MCTS pour obtenir le meilleur coup et la distribution de probabilité des coups
            move, policy_distribution, simulations = mcts_search(board, model)
            
            # Enregistrez les simulations ici
            simulations_data.append(simulations)

            # Stocker la position, la distribution de MCTS et le résultat temporaire (0)
            # `board_to_tensor` convertit la position du plateau en tenseur pour l'entraînement
            game_data.append((board_to_tensor(board), policy_distribution, 0))
            
            # Jouer le meilleur coup
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


    # Train DeepNN avec les données de self-play qui ont été accumulées et qui permettent d'entraîner le réseau

    def train(self):
        self.model.train()
        
        for epoch in range(self.epochs):  
            if len(self.self_play_data) < self.batch_size:
                continue  # Vérification pour s'assurer qu'on a suffisamment de données
            
            # Tirer un batch de données aléatoirement
            batch = random.sample(self.self_play_data, self.batch_size)
            states, policy_targets, value_targets = zip(*batch) # Diviser les données en trois listes différentes pour les états, les cibles de politique et les cibles de valeur
            
            # Préparer les tenseurs
            states = torch.tensor(states, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            
            # Prédictions du modèle
            predicted_policy, predicted_value = self.model(states)
            
            # Calcul des pertes
            policy_loss = -torch.sum(policy_targets * F.log_softmax(predicted_policy, dim=1)) / self.batch_size
            value_loss = F.mse_loss(predicted_value.view(-1), value_targets)
            loss = policy_loss + value_loss
            
            # Backpropagation et mise à jour des poids
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Afficher la perte pour suivre l'entraînement
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Policy Loss: {policy_loss.item():.4f} | Value Loss: {value_loss.item():.4f}")

    def run_training_loop(self, num_games=10000):
        """Exécute un cycle complet de self-play et d'entraînement."""
        for game in range(num_games):
            print(f"--- Partie {game + 1} ---")
            self.play_game()  # Générer une nouvelle partie via self-play
            self.train()      # Entraîner le modèle avec les données de self-play
            print(f"Fin de la partie {game + 1}")

    def save_simulations(self, simulations_data):
        """Sauvegarde les simulations dans un fichier JSON."""
        with open("simulations.json", "a") as file:
            json.dump(simulations_data, file)
            file.write("\n")
        print(f"Simulations enregistrées pour cette partie.")

