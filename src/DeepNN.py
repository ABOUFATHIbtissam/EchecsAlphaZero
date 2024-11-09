import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        
        #Couches Convolutionnelles Initiales
        self.conv1 = nn.Conv2d(12, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        #Residual layer permet de ajouter les entrées de la couche précédente à la sortie de la couche actuelle 
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256), #batch normalization
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)
            ) for _ in range(20)  #20 residual blocks /// ou 40 à voir
        ])
        
        #The policy head pour la distribution de probabilité sur les coups possibles
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, 467)  #2 canaux de sortie, 467 : approximation nombre de coups possibles
        
        #The value head qui estime la probabilité de gagner la partie à partir de l'état actuel
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256) # 8*8 : taille de l'échiquier, 256 : taille de la sortie de la couche
        self.value_fc2 = nn.Linear(256, 1) #redution de la taille de la sortie à 1
        
    def forward(self, x):
        # Couches de convolution
        x = F.relu(self.conv1(x))
        
        # Passer par les blocs résiduels
        for block in self.residual_blocks:
            residual = x
            x = block(x) + residual
            x = F.relu(x)
        
        #The policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(-1, 2 * 8 * 8) # transformer le tenseur en un vecteur de taille 2*8*8 = 128
        policy = F.log_softmax(self.policy_fc(policy), dim=1) #log-softmax pour les probas des coups
        
        #The value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value)) #une valeur de sortie entre -1 et 1
        
        return policy, value
