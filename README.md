# EchecsAlphaGoZero

## Prérequis
Assurez-vous d'avoir les éléments suivants installés sur votre machine :

- Python 3.6 ou supérieur
- pip
- Pygame
- PyTorch
- python-chess
- NumPy

## Installation
1. Clonez le dépôt sur votre machine locale :

    ```sh
    git clone https://github.com/ABOUFATHIbtissam/EchecsAlphaGoZero.git
    cd EchecsAlphaGoZero
    ```

## Utilisation

1. Pour lancer une partie d'échecs entre deux IA :

    ```sh
    cd src
    python Main.py
    ```

2. Pour lancer l'entraînement du modèle :

    ```sh
    cd src
    python Train.py
    ```

## Structure du projet

- `src/DeepNN.py` : Contient la définition du modèle de réseau neuronal.
- `src/TransformTensor.py` : Contient la fonction pour convertir un plateau d'échecs en tenseur.
- `src/MCTS.py` : Contient l'implémentation de la recherche Monte Carlo Tree Search.
- `src/Partie.py` : Contient la classe pour gérer une partie d'échecs.
- `src/Train.py` : Contient la classe et les fonctions pour l'entraînement du modèle.
- `src/Main.py` : Point d'entrée principal pour lancer le jeu ou l'entraînement.
