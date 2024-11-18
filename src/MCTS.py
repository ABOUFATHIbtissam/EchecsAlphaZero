import random
import chess
import torch
import math
from TransformTensor import board_to_tensor

class MCTS:
    def __init__(self, board, parent=None, prob=0.0):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0 #nombre de fois que le noeud a été visité
        self.value_sum = 0.0 #somme des valeurs des noeuds visités
        self.prob = prob  #probabilité de politique associée au noeud
    
    def value(self):
        """Calcule la valeur moyenne du nœud."""
        if self.visits == 0:
            return 0  #la valeur initiale pour les noeuds non visités
        return self.value_sum / self.visits

    def uct_value(self, c=1.4):
        """Calcule la valeur UCT pour le noeud."""
        if self.visits == 0:
            return float('inf')  #pour favoriser l'exploration des nœuds non visités
        exploration_term = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return self.value() + exploration_term

    def expand(self, policy_distribution):
        """Crée de nouveaux noeuds enfants pour chaque coup légal depuis l'état actuel."""
        moves = list(self.board.legal_moves)
        for move, proba in zip(moves, policy_distribution):
            new_board = self.board.copy() 
            new_board.push(move)
            self.children.append(MCTS(new_board, parent=self, prob=proba)) #ajouter le noeud fils
    
    def best_child(self, c=1.4):
        """Sélectionne le meilleur enfant en fonction de la valeur UCT"""
        return max(self.children, key=lambda node: node.uct_value(c)) 
    
    def is_leaf(self):
        """Vérifie si le nœud est une feuille."""
        return len(self.children) == 0 

    def most_visited_child(self):
        """Retourne l'enfant le plus visité, pour la sélection du mouvement final."""
        return max(self.children, key=lambda node: node.visits)


def mcts_search(board, model, simulations=50):
    root = MCTS(board)
    simulation_count = 0
    if board.is_game_over():
        raise print("Recherche MCTS appelée sur un plateau terminé.")
    
    for _ in range(simulations):
        simulation_count += 1
        node = root
        #Descendre dans l'arbre
        while not node.is_leaf() and not node.board.is_game_over():
            node = node.best_child()
        
        #Expansion
        if not node.board.is_game_over():
            #Evaluer la valeur du noeud
            state_tensor = torch.tensor(board_to_tensor(node.board)).unsqueeze(0).permute(0, 3, 1, 2)
            policy, value = model(state_tensor)
            
            #Convertir la politique en distribution de probabilité
            policy_distribution = torch.softmax(policy, dim=1).squeeze().detach().numpy()

            node.expand(policy_distribution)
        
        #Évaluation
        state_tensor = torch.tensor(board_to_tensor(node.board)).unsqueeze(0).permute(0, 3, 1, 2)
        _, value = model(state_tensor)
       
        #Remonter et propager la valeur du noeud
        while node.parent:
            node = node.parent
            node.value_sum += value.item()
            node.visits += 1

    # Retourner le meilleur coup basé sur la politique calculée
    best_move = root.best_child().board.peek()
    
    return best_move, policy_distribution, [child.visits for child in root.children], simulation_count

