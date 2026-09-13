import numpy as np

# Definindo dois vetores de exemplo
vetor_A = np.array([1.0, 2.0, 3.0])
vetor_B = np.array([4.0, 6.0, 8.0])

# Cálculo da distância euclidiana
distancia = np.linalg.norm(vetor_A - vetor_B)
print(f"Distância Euclidiana: {distancia}")