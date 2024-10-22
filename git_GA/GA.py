import numpy as np
import random

# Definiamo la funzione che simula la dinamica del uniciclo
def uniciclo_dinamica(x, y, theta, v, omega, dt):
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt
    theta_next = theta + omega * dt
    return x_next, y_next, theta_next

# Funzione di simulazione MPC
def simulazione_mpc(params, target, T, dt=0.1):
    # Parametri del controllo da ottimizzare
    P_diag = params[:3]  # 3 elementi per la matrice P
    Q_diag = params[3:5] # 2 elementi per la matrice Q
    horizon = int(params[5]) # Orizzonte temporale (integer)
    
    P = np.diag(P_diag)
    Q = np.diag(Q_diag)
    
    # Stato iniziale
    x, y, theta = 0, 0, 0  # Iniziamo dal punto (0, 0, 0)
    
    total_cost = 0
    
    for _ in range(int(T/dt)):
        # Errore tra lo stato attuale e il target
        state_error = np.array([x - target[0], y - target[1], theta - target[2]])
        
        # Generazione di controlli ottimali (qui semplice esempio proporzionale)
        v = -Q_diag[0] * state_error[0]  # Controllo su v basato sull'errore di x
        omega = -Q_diag[1] * state_error[2]  # Controllo su omega basato sull'errore di theta
        
        # Costruzione della funzione di costo (errore di stato e costo dei controlli)
        state_cost = state_error.T @ P @ state_error
        control_cost = v**2 * Q_diag[0] + omega**2 * Q_diag[1]
        total_cost += state_cost + control_cost
        
        # Aggiorna lo stato
        x, y, theta = uniciclo_dinamica(x, y, theta, v, omega, dt)
        
        # Condizione di termine anticipato (se raggiunto l'obiettivo)
        if np.linalg.norm([x - target[0], y - target[1]]) < 0.1 and abs(theta - target[2]) < 0.1:
            break
    
    return total_cost

# Funzione fitness per l'algoritmo genetico
def fitness(params, target, T):
    cost = simulazione_mpc(params, target, T)
    return 1 / (1 + cost)  # Invertiamo il costo per trasformarlo in fitness

# Algoritmo genetico
def algoritmo_genetico(pop_size, generations, target, T):
    # Inizializzazione della popolazione (6 parametri)
    population = [np.random.rand(6) for _ in range(pop_size)]
    
    for gen in range(generations):
        # Calcola la fitness per ogni individuo
        fitness_scores = [fitness(ind, target, T) for ind in population]
        
        # Seleziona i migliori (elitismo)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        population = [population[i] for i in sorted_indices[:pop_size // 2]]
        
        # Crossover e mutazione
        while len(population) < pop_size:
            # Seleziona due genitori
            parent1, parent2 = random.sample(population[:pop_size // 4], 2)
            # Crossover
            crossover_point = random.randint(1, 5)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            # Mutazione
            if random.random() < 0.1:  # probabilità di mutazione
                mutation_idx = random.randint(0, 5)
                child[mutation_idx] = np.random.rand()
            population.append(child)
        
        # Stampa il progresso
        best_fitness = max(fitness_scores)
        print(f"Generazione {gen + 1}, Migliore fitness: {best_fitness:.4f}")
    
    # Ritorna l'individuo migliore
    best_individual = population[0]
    return best_individual

# Parametri di simulazione
target = [10, 10, np.pi/2]  # Obiettivo finale (x, y, theta)
T = 10  # Tempo totale di simulazione
pop_size = 20  # Numero di individui nella popolazione
generations = 50  # Numero di generazioni

# Esecuzione dell'algoritmo genetico
best_params = algoritmo_genetico(pop_size, generations, target, T)
print(f"Migliori parametri trovati: {best_params}")