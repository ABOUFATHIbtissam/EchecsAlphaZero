import random
import chess
import torch
from TransformTensor import board_to_tensor

class MCTS:
    def __init__(self, board, parent=None):
        self.board = board
        self.parent = parent
        self.children = []
        self.visits = 0 #nombre de fois que le noeud a été visité
        self.value_sum = 0.0 #somme des valeurs des noeuds visités
    
    def value(self):
        if self.visits == 0:
            return float('inf')  #pour favoriser l'exploration des noeuds non visités
        return self.value_sum / self.visits #valeur moyenne des noeuds visités

    def expand(self):
        moves = list(self.board.legal_moves) 
        for move in moves:
            new_board = self.board.copy() 
            new_board.push(move) #effectuer le coup
            self.children.append(MCTS(new_board, parent=self)) #ajouter le noeud fils
    
    def best_child(self):
        return max(self.children, key=lambda node: node.value()) #retourner le noeud fils avec la meilleure valeur
    
    def is_leaf(self):
        return len(self.children) == 0 #retourner True si le noeud est une feuille

def mcts_search(board, model, simulations=50):
    root = MCTS(board)
    simulation_count = 0
    
    for _ in range(simulations):
        simulation_count += 1
        node = root
        # Descendre dans l'arbre
        while node.children:
            node = node.best_child()
        
        # Expansion
        if not node.board.is_game_over():
            node.expand()
        
        # Evaluer la valeur du noeud
        state_tensor = torch.tensor(board_to_tensor(node.board)).unsqueeze(0).permute(0, 3, 1, 2)
        policy, value = model(state_tensor)
        
        # Normaliser la politique
        policy_distribution = torch.softmax(policy, dim=1).squeeze().detach().numpy()

        node.value_sum += value.item()
        node.visits += 1
        
        # Remonter et propager la valeur du noeud
        while node.parent:
            node = node.parent
            node.value_sum += value.item()
            node.visits += 1

    # Retourner le meilleur coup basé sur la politique calculée
    best_move = root.best_child().board.peek()
    
    return best_move, policy_distribution, simulation_count
