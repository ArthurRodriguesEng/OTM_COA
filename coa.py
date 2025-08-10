import numpy as np
import matplotlib.pyplot as plt
import time

try:
    from fob_func import fob
except ImportError:
    print("Erro: O arquivo 'fob_func.pyd' não foi encontrado.")
    print("Certifique-se de que o arquivo está na mesma pasta e renomeado corretamente.")
    exit()

def load_initial_population(filepath):
    """
    Lê a população inicial e os parâmetros de um arquivo de texto.

    Args:
        filepath (str): O caminho para o arquivo de texto.

    Returns:
        tuple: Um array numpy com a população inicial e o número máximo de iterações.
    """
    initial_pop = []
    max_iter = None
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line or "VALORES" in line:
                continue
            
            if "ITER_MAX" in line:
                # Extrai o valor de ITER_MAX
                max_iter = int(line.split('=')[1].replace(';', ''))
            else:
                # Extrai as coordenadas X e Y
                parts = line.split()
                if len(parts) == 2:
                    initial_pop.append([float(parts[0]), float(parts[1])])

        if max_iter is None:
            raise ValueError("A variável 'ITER_MAX' não foi encontrada no arquivo.")
        if not initial_pop:
            raise ValueError("Nenhum dado de população inicial foi encontrado no arquivo.")
            
        return np.array(initial_pop), max_iter
    except FileNotFoundError:
        print(f"Erro: Arquivo de população inicial '{filepath}' não encontrado.")
        exit()
    except Exception as e:
        print(f"Erro ao processar o arquivo '{filepath}': {e}")
        exit()


def coati_optimization_algorithm(objective_func, bounds, n_population, max_iter, initial_population=None):
    """
    Implementação do Coati Optimization Algorithm (COA) com suporte para população inicial.
    """
    dim = len(bounds)
    lb = np.array([b[0] for b in bounds])
    ub = np.array([b[1] for b in bounds])

    # 1. Inicialização
    if initial_population is not None:
        population = np.array(initial_population)
        # Ajusta o tamanho da população com base no arquivo fornecido
        n_population = population.shape[0]
    else:
        # Geração aleatória padrão se nenhuma população for fornecida
        population = lb + np.random.rand(n_population, dim) * (ub - lb)

    fitness = np.array([objective_func(ind.tolist()) for ind in population])

    best_idx = np.argmin(fitness)
    iguana_pos = population[best_idx].copy()
    iguana_fitness = fitness[best_idx]
    
    convergence_curve = []

    # 2. Loop de Iterações
    for t in range(1, max_iter + 1):
        # Fase 1: Exploração
        # Primeira metade
        for i in range(n_population // 2):
            r, I = np.random.rand(dim), np.random.randint(1, 3)
            new_pos = np.clip(population[i] + r * (iguana_pos - I * population[i]), lb, ub)
            new_fitness = objective_func(new_pos.tolist())
            if new_fitness < fitness[i]:
                population[i], fitness[i] = new_pos, new_fitness

        # Iguana "cai"
        iguana_ground_pos = lb + np.random.rand(dim) * (ub - lb)
        iguana_ground_fitness = objective_func(iguana_ground_pos.tolist())

        # Segunda metade
        for i in range(n_population // 2, n_population):
            r, I = np.random.rand(dim), np.random.randint(1, 3)
            if iguana_ground_fitness < fitness[i]:
                new_pos = population[i] + r * (iguana_ground_pos - I * population[i])
            else:
                new_pos = population[i] + r * (population[i] - iguana_ground_pos)
            new_pos = np.clip(new_pos, lb, ub)
            new_fitness = objective_func(new_pos.tolist())
            if new_fitness < fitness[i]:
                population[i], fitness[i] = new_pos, new_fitness
        
        # Fase 2: Explotação
        for i in range(n_population):
            local_lb, local_ub = lb / t, ub / t
            r = np.random.rand(dim)
            new_pos = population[i] + (1 - 2 * r) * (local_lb + r * (local_ub - local_lb))
            new_pos = np.clip(new_pos, lb, ub)
            new_fitness = objective_func(new_pos.tolist())
            if new_fitness < fitness[i]:
                population[i], fitness[i] = new_pos, new_fitness

        # Atualiza a melhor solução global (Iguana)
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < iguana_fitness:
            iguana_fitness = fitness[current_best_idx]
            iguana_pos = population[current_best_idx].copy()
        
        convergence_curve.append(iguana_fitness)
        
    return iguana_pos, iguana_fitness, convergence_curve

# --- EXECUÇÃO PRINCIPAL E ANÁLISE ---
if __name__ == "__main__":
    
    # Carrega os dados do arquivo
    

    initial_pop_filepath = 'populacao_incial.txt'
    # initial_population, max_iter = load_initial_population(initial_pop_filepath)
    # n_population = len(initial_population)
    # print(f"Utilizando população inicial do arquivo: '{initial_pop_filepath}'")
    
    n_population = 100  # Tamanho da população (definido pelo arquivo de entrada)
    max_iter = 10000   # Número máximo de iterações (definido] pelo arquivo de entrada)
    initial_population = None  # População inicial não utilizada neste exemplo
    
    #Posição (Valores de X e Y): [420.968490, 420.968394]
    
    # Define os limites do problema
    bounds = [(-500, 500), (-500, 500)]

    print("=" * 70)
    print("Iniciando Otimização com o Algoritmo Coati (COA)")
    print("=" * 70)

    start_time = time.time()
    
    # Executa o algoritmo com a população inicial
    best_pos, best_fit, curve = coati_optimization_algorithm(
        objective_func=fob,
        bounds=bounds,
        n_population=n_population,
        max_iter=max_iter,
        initial_population=initial_population
    )
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("RELATÓRIO DA EXECUÇÃO")
    print("="*70)
    
    # 1. A solução encontrada
    print("\n--- 1. Solução Encontrada ---")
    print(f"  - Posição (Valores de X e Y): [{best_pos[0]:.6f}, {best_pos[1]:.6f}]")
    print(f"  - Valor da Função Objetivo (Mínimo): {best_fit:.6f}")
    
    # 2. O método utilizado
    print("\n--- 2. Método Utilizado ---")
    print("  - Algoritmo: Coati Optimization Algorithm (COA).")
    print("  - Parâmetros da Execução:")
    print(f"    - Tamanho da População (N): {n_population} (definido pelo arquivo de entrada)")
    print(f"    - Número Máximo de Iterações (T): {max_iter} (definido pelo arquivo de entrada)")
    
    # 3. Visualização da evolução
    print("\n--- 3. Visualização da Evolução ---")
    plt.figure(figsize=(12, 6))
    plt.plot(curve, color='red', linewidth=2)
    plt.title("Evolução da Função Objetivo (Melhor Valor por Iteração)")
    plt.xlabel("Iteração")
    plt.ylabel("Valor da Função Objetivo")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig("convergencia_populacao_inicial.png")
    print("  - Gráfico 'convergencia_populacao_inicial.png' salvo com sucesso.")
    
    print(f"\nTempo total de execução: {total_time:.2f} segundos.")
    print("\n" + "="*70)