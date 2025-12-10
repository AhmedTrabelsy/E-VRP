import os
import random
import math
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# =========================
# Lecture du dataset
# =========================
def read_instance(filepath):
    nodes = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    header = None
    for i, line in enumerate(lines):
        if line.strip().startswith("StringID"):
            header = i
            break
    if header is None:
        raise ValueError("En-tête non trouvée dans le fichier.")
    for line in lines[header+1:]:
        parts = line.strip().split()
        if len(parts) < 11 or parts[1] not in ('c', 'd'):
            continue
        node = {
            'id': parts[0],
            'type': parts[1],
            'x': float(parts[2]),
            'y': float(parts[3]),
            'demand': float(parts[4]),
            'delivery_demand': float(parts[5]),
            'pickup_demand': float(parts[6]),
            'division_rate': float(parts[7]),
            'ready_time': float(parts[8]),
            'due_date': float(parts[9]),
            'service_time': float(parts[10])
        }
        nodes.append(node)
    nodes.sort(key=lambda n: 0 if n['id'][0] == 'D' else 1)
    return nodes

def distance(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])

# =========================
# Fonction objectif
# =========================
def objective(solution, nodes, vehicle_capacity, battery_capacity):
    total_distance = 0
    total_cost = 0
    total_delay_penalty = 0
    clients_served = set()
    for route in solution:
        if not route:
            continue
        prev = nodes[0]
        load = 0
        battery = battery_capacity
        time = prev['ready_time']
        for idx in route:
            node = nodes[idx]
            dist = distance(prev, node)
            total_distance += dist
            battery -= dist * node['division_rate']
            load += node['demand']
            time += dist
            if time < node['ready_time']:
                time = node['ready_time']
            if time > node['due_date']:
                total_delay_penalty += (time - node['due_date'])
            time += node['service_time']
            prev = node
            if node['type'] == 'c':
                clients_served.add(idx)
        dist = distance(prev, nodes[0])
        total_distance += dist
        battery -= dist * nodes[0]['division_rate']
        if load > vehicle_capacity or battery < 0:
            total_cost += 10000
    return total_distance + total_cost + total_delay_penalty - 100 * len(clients_served)

# =========================
# Génération initiale
# =========================
def generate_initial_solution(nodes, vehicle_capacity):
    customers = [i for i in range(1, len(nodes)) if nodes[i]['type'] == 'c']
    random.shuffle(customers)
    solution = []
    route = []
    load = 0
    for idx in customers:
        demand = nodes[idx]['demand']
        if load + demand > vehicle_capacity:
            solution.append(route)
            route = []
            load = 0
        route.append(idx)
        load += demand
    if route:
        solution.append(route)
    return solution

# =========================
# GA opérateurs
# =========================
def selection(population, fitness):
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitness[i] < fitness[j] else population[j]

def crossover(parent1, parent2):
    if not parent1 or not parent2:
        return parent1[:], parent2[:]
    cut1 = random.randint(0, len(parent1)-1)
    cut2 = random.randint(0, len(parent2)-1)
    child1 = parent1[:cut1] + parent2[cut2:]
    child2 = parent2[:cut2] + parent1[cut1:]
    return child1, child2

def mutate(solution, nodes):
    new_solution = [route[:] for route in solution]
    customers = [idx for route in new_solution for idx in route]
    if len(customers) < 2:
        return new_solution
    idx1, idx2 = random.sample(customers, 2)
    r1 = r2 = None
    for i, route in enumerate(new_solution):
        if idx1 in route:
            r1 = i
        if idx2 in route:
            r2 = i
    if r1 is not None and r2 is not None:
        i1 = new_solution[r1].index(idx1)
        i2 = new_solution[r2].index(idx2)
        new_solution[r1][i1], new_solution[r2][i2] = new_solution[r2][i2], new_solution[r1][i1]
    return new_solution

# =========================
# GA principal
# =========================
def GA_algorithm(nodes, vehicle_capacity, battery_capacity,
                 pop_size=30, max_gen=100, crossover_rate=0.8, mutation_rate=0.2):
    population = [generate_initial_solution(nodes, vehicle_capacity) for _ in range(pop_size)]
    fitness = [objective(sol, nodes, vehicle_capacity, battery_capacity) for sol in population]

    for gen in range(max_gen):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = selection(population, fitness)
            parent2 = selection(population, fitness)
            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            if random.random() < mutation_rate:
                child1 = mutate(child1, nodes)
            if random.random() < mutation_rate:
                child2 = mutate(child2, nodes)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]
        fitness = [objective(sol, nodes, vehicle_capacity, battery_capacity) for sol in population]

    best_idx = fitness.index(min(fitness))
    return population[best_idx], fitness[best_idx]

# =========================
# Visualisation avec streamlit-agraph
# =========================
def plot_solution_agraph(solution, nodes):
    node_objs = []
    edge_objs = []

    # Ajout du dépôt
    depot = nodes[0]
    node_objs.append(Node(id=depot['id'], label="Depot", size=30, color="red"))

    # Ajout des clients
    for n in nodes[1:]:
        color = "blue" if n['type'] == 'c' else "green"
        node_objs.append(Node(id=n['id'], label=n['id'], size=20, color=color))

    # Ajout des routes
    for i, route in enumerate(solution):
        prev = depot['id']
        for idx in route:
            edge_objs.append(Edge(source=prev, target=nodes[idx]['id'], color="orange"))
            prev = nodes[idx]['id']
        edge_objs.append(Edge(source=prev, target=depot['id'], color="orange"))

    config = Config(width=800, height=600, directed=True, physics=True, hierarchical=False)
    return agraph(nodes=node_objs, edges=edge_objs, config=config)

# =========================
# Streamlit UI
# =========================
st.title("🚚 Algorithme Génétique pour 2E-EVRP")

with st.sidebar:
    st.header("⚙️ Paramètres")
    base_path = "./2E-EVRP-Instances-v2"
    type_choice = st.selectbox("Type :", ["Type_x", "Type_y"])
    customer_choice = st.selectbox("Nombre de clients :", ["Customer_5", "Customer_10", "Customer_15", "Customer_50", "Customer_100"])
    folder = os.path.join(base_path, type_choice, customer_choice)
    files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    instance_choice = st.selectbox("Instance :", files)

    vehicle_capacity = st.number_input("Capacité véhicule", value=250.0)
    battery_capacity = st.number_input("Capacité batterie", value=194.6)

    pop_size = st.slider("Population size", 10, 100, 30)
    max_gen = st.slider("Generations", 10, 500, 100)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.8)
    mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.2)

if st.sidebar.button("🚀 Lancer GA"):
    filepath = os.path.join(folder, instance_choice)
    nodes = read_instance(filepath)

    # Exécution de l'algorithme génétique
    best_solution, best_cost = GA_algorithm(
        nodes,
        vehicle_capacity,
        battery_capacity,
        pop_size=pop_size,
        max_gen=max_gen,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )

    # Résultats
    st.subheader(f"✅ Résultats pour {instance_choice}")
    st.write(f"**Score optimisé :** {best_cost:.2f}")
    st.write(f"**Nombre de tournées :** {len(best_solution)}")

    # Visualisation interactive avec streamlit-agraph
    st.subheader("📊 Visualisation des tournées")
    plot_solution_agraph(best_solution, nodes)