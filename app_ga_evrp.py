# streamlit_app_with_stations.py
import os
import random
import math
import re
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config

# -------------------------
# Lecture du dataset (d, c, f, s)
# -------------------------


def extract_number_from_slashes(line):
    parts = line.split('/')
    for p in reversed(parts):
        p = p.strip()
        if not p:
            continue
        try:
            return float(p)
        except:
            m = re.search(r'[-+]?\d*\.\d+|\d+', p)
            if m:
                try:
                    return float(m.group())
                except:
                    continue
    return None


def read_instance(filepath):
    nodes = []
    L = C = Q = None
    r = g = v = None

    with open(filepath, "r") as f:
        raw_lines = f.readlines()
    lines = [ln.rstrip("\n") for ln in raw_lines]

    header = None
    for i, line in enumerate(lines):
        if line.strip().startswith("StringID"):
            header = i
            break
    if header is None:
        raise ValueError("Header 'StringID' not found in instance file.")

    param_prefixes = ("L ", "C ", "Q ", "r ", "g ", "v ")
    for line in lines[header+1:]:
        if not line.strip():
            continue
        if any(line.lstrip().startswith(pref) for pref in param_prefixes):
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        node_type = parts[1]
        if node_type not in ('d', 'c', 'f', 's'):
            continue
        try:
            node = {
                'id': parts[0],
                'type': node_type,
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
        except:
            continue
        nodes.append(node)

    for line in lines:
        l = line.strip()
        if l.startswith("L "):
            val = extract_number_from_slashes(line)
            if val is not None:
                L = val
        elif l.startswith("C "):
            val = extract_number_from_slashes(line)
            if val is not None:
                C = val
        elif l.startswith("Q "):
            val = extract_number_from_slashes(line)
            if val is not None:
                Q = val
        elif l.startswith("r "):
            val = extract_number_from_slashes(line)
            if val is not None:
                r = val
        elif l.startswith("g "):
            val = extract_number_from_slashes(line)
            if val is not None:
                g = val
        elif l.startswith("v "):
            val = extract_number_from_slashes(line)
            if val is not None:
                v = val

    # fallback
    if L is None:
        for line in lines:
            if line.startswith("L"):
                m = re.search(r'[-+]?\d*\.\d+|\d+', line)
                if m:
                    L = float(m.group())
                    break
    if C is None:
        for line in lines:
            if line.startswith("C"):
                m = re.search(r'[-+]?\d*\.\d+|\d+', line)
                if m:
                    C = float(m.group())
                    break
    if Q is None:
        for line in lines:
            if line.startswith("Q"):
                m = re.search(r'[-+]?\d*\.\d+|\d+', line)
                if m:
                    Q = float(m.group())
                    break
    if r is None:
        for line in lines:
            if line.startswith("r"):
                m = re.search(r'[-+]?\d*\.\d+|\d+', line)
                if m:
                    r = float(m.group())
                    break
    if g is None:
        for line in lines:
            if line.startswith("g"):
                m = re.search(r'[-+]?\d*\.\d+|\d+', line)
                if m:
                    g = float(m.group())
                    break
    if v is None:
        for line in lines:
            if line.startswith("v"):
                m = re.search(r'[-+]?\d*\.\d+|\d+', line)
                if m:
                    v = float(m.group())
                    break

    if None in (L, C, Q, r, g, v):
        missing = [name for name, val in (
            ("L", L), ("C", C), ("Q", Q), ("r", r), ("g", g), ("v", v)) if val is None]
        raise ValueError(
            f"Missing global parameter(s) in instance file: {', '.join(missing)}")

    nodes.sort(key=lambda n: 0 if n['type'] == 'd' else (
        1 if n['type'] == 's' else (2 if n['type'] == 'f' else 3)))

    return nodes, L, C, Q, r, g, v

# -------------------------
# Distances utilitaires
# -------------------------


def distance(a, b):
    return math.hypot(a['x'] - b['x'], a['y'] - b['y'])


def find_nearest_recharge(current_idx, nodes, recharge_indices):
    if not recharge_indices:
        return None
    cur = nodes[current_idx]
    best = min(recharge_indices, key=lambda s: distance(cur, nodes[s]))
    return best


def find_reachable_recharge(current_idx, nodes, recharge_indices, battery):
    if not recharge_indices:
        return None
    cur = nodes[current_idx]
    reachable = []
    for s in recharge_indices:
        d = distance(cur, nodes[s])
        consumption = d * nodes[s]['division_rate']
        if consumption <= battery:
            reachable.append((s, d))
    if reachable:
        return min(reachable, key=lambda x: x[1])[0]
    return find_nearest_recharge(current_idx, nodes, recharge_indices)

# -------------------------
# Séquence → Routes avec logique satellite
# -------------------------


def sequence_to_routes(sequence, nodes, vehicle_capacity, battery_capacity, recharge_indices):
    routes = []
    route = []
    load = 0.0
    battery = battery_capacity
    prev_idx = 0
    threshold_load_for_satellite = vehicle_capacity * \
        0.7  # visit satellite only if load >= 70%

    for cust_idx in sequence:
        cust = nodes[cust_idx]

        # si capacité dépassée → nouvelle route
        if load + cust['demand'] > vehicle_capacity:
            if route:
                routes.append(route)
            route = []
            load = 0.0
            battery = battery_capacity
            prev_idx = 0

        dist_to_c = distance(nodes[prev_idx], cust)
        battery_needed = dist_to_c * cust['division_rate']

        # Vérifier batterie
        if battery_needed > battery:
            # choisir recharge reachable
            recharge = find_reachable_recharge(
                prev_idx, nodes, recharge_indices, battery)
            if recharge is not None and (len(route) == 0 or route[-1] != recharge):
                # logic: f=station, s=satellite only if load >= threshold
                node_recharge = nodes[recharge]
                if node_recharge['type'] == 'f' or (node_recharge['type'] == 's' and load >= threshold_load_for_satellite):
                    if prev_idx != recharge:
                        route.append(recharge)
                        battery = battery_capacity
                        prev_idx = recharge

        route.append(cust_idx)
        battery -= distance(nodes[prev_idx], cust) * cust['division_rate']
        load += cust['demand']
        prev_idx = cust_idx

    if route:
        routes.append(route)
    return routes

# -------------------------
# Génération initiale
# -------------------------


def generate_initial_solution(nodes, vehicle_capacity, battery_capacity):
    customer_indices = [i for i in range(
        1, len(nodes)) if nodes[i]['type'] == 'c']
    recharge_indices = [i for i in range(
        1, len(nodes)) if nodes[i]['type'] in ('f', 's')]
    random.shuffle(customer_indices)
    return sequence_to_routes(customer_indices, nodes, vehicle_capacity, battery_capacity, recharge_indices)

# -------------------------
# GA: sélection, crossover, mutation
# -------------------------


def selection(population, fitness):
    i, j = random.sample(range(len(population)), 2)
    return population[i] if fitness[i] < fitness[j] else population[j]


def crossover(parent1, parent2, nodes, vehicle_capacity, battery_capacity):
    seq1 = [i for r in parent1 for i in r if nodes[i]['type'] == 'c']
    seq2 = [i for r in parent2 for i in r if nodes[i]['type'] == 'c']
    if not seq1 or not seq2:
        return parent1[:], parent2[:]
    a, b = sorted(random.sample(range(len(seq1)), 2))
    seg = seq1[a:b+1]

    def ox(source, donor_seg):
        child = [x for x in source if x not in donor_seg]
        mid = len(child)//2
        return child[:mid] + donor_seg + child[mid:]

    child_seq1 = ox(seq2, seg)
    a2, b2 = sorted(random.sample(range(len(seq2)), 2))
    seg2 = seq2[a2:b2+1]
    child_seq2 = ox(seq1, seg2)
    recharge_indices = [i for i in range(
        1, len(nodes)) if nodes[i]['type'] in ('f', 's')]
    return (sequence_to_routes(child_seq1, nodes, vehicle_capacity, battery_capacity, recharge_indices),
            sequence_to_routes(child_seq2, nodes, vehicle_capacity, battery_capacity, recharge_indices))


def mutate(solution, nodes, vehicle_capacity, battery_capacity):
    clients = [i for r in solution for i in r if nodes[i]['type'] == 'c']
    if len(clients) < 2:
        return solution
    a, b = random.sample(range(len(clients)), 2)
    clients[a], clients[b] = clients[b], clients[a]
    recharge_indices = [i for i in range(
        1, len(nodes)) if nodes[i]['type'] in ('f', 's')]
    return sequence_to_routes(clients, nodes, vehicle_capacity, battery_capacity, recharge_indices)

# -------------------------
# Simulation et objectif
# -------------------------


def simulate_route(route, nodes, vehicle_capacity, battery_capacity):
    total_distance = 0.0
    total_delay_penalty = 0.0
    total_cost = 0.0
    clients_served = set()
    visits = []
    prev = nodes[0]
    battery = battery_capacity
    load = 0.0
    time = 0.0
    for idx in route:
        node = nodes[idx]
        dist = distance(prev, node)
        total_distance += dist
        battery -= dist * node['division_rate']
        time += dist
        if time < node['ready_time']:
            time = node['ready_time']
        if time > node['due_date']:
            total_delay_penalty += (time - node['due_date'])
        time += node['service_time']
        load += node['demand']
        if load > vehicle_capacity:
            total_cost += 10000
        if node['type'] in ('f', 's'):
            battery = battery_capacity
        if battery < 0:
            total_cost += 10000
        battery_pct = max(
            0.0, min(100.0, (battery / battery_capacity) * 100.0))
        visits.append((idx, battery_pct, load))
        if node['type'] == 'c':
            clients_served.add(idx)
        prev = node
    dist = distance(prev, nodes[0])
    total_distance += dist
    battery -= dist * nodes[0]['division_rate']
    if battery < 0:
        total_cost += 10000
    return {
        'distance': total_distance,
        'delay_penalty': total_delay_penalty,
        'cost': total_cost,
        'clients_served': clients_served,
        'visits': visits
    }


def objective(solution, nodes, vehicle_capacity, battery_capacity):
    total_distance = 0.0
    total_cost = 0.0
    total_delay_penalty = 0.0
    all_clients = set()
    per_route_visits = []
    for route in solution:
        sim = simulate_route(route, nodes, vehicle_capacity, battery_capacity)
        total_distance += sim['distance']
        total_delay_penalty += sim['delay_penalty']
        total_cost += sim['cost']
        all_clients.update(sim['clients_served'])
        per_route_visits.append(sim['visits'])
    score = total_distance + total_cost + \
        total_delay_penalty - 100.0 * len(all_clients)
    return score, {
        'distance': total_distance,
        'penalty_cost': total_cost,
        'delay_penalty': total_delay_penalty,
        'clients_served': len(all_clients),
        'per_route_visits': per_route_visits
    }

# -------------------------
# GA main
# -------------------------


def GA_algorithm(nodes, vehicle_capacity, battery_capacity,
                 pop_size=30, max_gen=100, crossover_rate=0.8, mutation_rate=0.2):
    population = [generate_initial_solution(
        nodes, vehicle_capacity, battery_capacity) for _ in range(pop_size)]
    fitness = []
    aux_info = []
    for sol in population:
        f, info = objective(sol, nodes, vehicle_capacity, battery_capacity)
        fitness.append(f)
        aux_info.append(info)
    for gen in range(max_gen):
        new_population = []
        while len(new_population) < pop_size:
            parent1 = selection(population, fitness)
            parent2 = selection(population, fitness)
            if random.random() < crossover_rate:
                child1, child2 = crossover(
                    parent1, parent2, nodes, vehicle_capacity, battery_capacity)
            else:
                child1, child2 = parent1[:], parent2[:]
            if random.random() < mutation_rate:
                child1 = mutate(
                    child1, nodes, vehicle_capacity, battery_capacity)
            if random.random() < mutation_rate:
                child2 = mutate(
                    child2, nodes, vehicle_capacity, battery_capacity)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]
        fitness = []
        aux_info = []
        for sol in population:
            f, info = objective(sol, nodes, vehicle_capacity, battery_capacity)
            fitness.append(f)
            aux_info.append(info)
    best_idx = fitness.index(min(fitness))
    return population[best_idx], fitness[best_idx], aux_info[best_idx]

# -------------------------
# Helpers affichage
# -------------------------


def get_display_name(node):
    raw = node['id']
    if node['type'] == 'c':
        return f"Client {raw[1:]}" if len(raw) > 1 else f"Client {raw}"
    if node['type'] == 's':
        return f"Satellite {raw[1:]}" if len(raw) > 1 else f"Satellite {raw}"
    if node['type'] == 'f':
        return f"Station {raw[1:]}" if len(raw) > 1 else f"Station {raw}"
    return f"Depot {raw[1:]}" if len(raw) > 1 else f"Depot {raw}"

# -------------------------
# Visualisation avec coûts sur edges
# -------------------------


def plot_solution_agraph(solution, nodes, per_route_visits, battery_capacity, vehicle_capacity):
    node_objs = []
    edge_objs = []
    depot = nodes[0]
    node_objs.append(Node(
        id=depot['id'],
        label="Dépôt",
        size=38,
        shape='image', image='https://icon-library.com/images/depot-icon/depot-icon-19.jpg',
        title=f"Dépôt ID: {depot['id']} | Coord: ({depot['x']}, {depot['y']})"
    ))
    for n in nodes[1:]:
        if n['type'] == 'c':
            label = get_display_name(n)
            image = 'https://cdn-icons-png.flaticon.com/512/7891/7891470.png'
            color = "#3498DB"
        elif n['type'] == 's':
            label = get_display_name(n)
            image = 'https://icon-library.com/images/depot-icon/depot-icon-19.jpg'
            color = "#9B59B6"
        elif n['type'] == 'f':
            label = get_display_name(n)
            image = 'https://cdn-icons-png.flaticon.com/512/9138/9138039.png'
            color = "#2ECC71"
        else:
            label = n['id']
            color = "gray"
        tooltip = (f"{label} | Type: {n['type']} | "
                   f"Coord: ({n['x']}, {n['y']}) | Demande: {n['demand']} | "
                   f"Ready: {n['ready_time']} | Due: {n['due_date']}")
        node_objs.append(Node(id=n['id'], label=label, size=24,
                         color=color, shape='image', image=image, title=tooltip))

    # edges avec coûts
    for route in solution:
        prev = depot['id']
        for idx in route:
            tgt = nodes[idx]['id']
            cost = distance(nodes[[n['id']
                            for n in nodes].index(prev)], nodes[idx])
            edge_objs.append(
                Edge(source=prev, target=tgt, color="#BDC3C7",
                     width=2, label=f"{cost:.0f}")
            )
            prev = tgt
        cost = distance(nodes[[n['id'] for n in nodes].index(prev)], depot)
        edge_objs.append(
            Edge(source=prev, target=depot['id'],
                 color="#BDC3C7", width=2, label=f"{cost:.0f}")
        )

    config = Config(width=1000, height=700, directed=True,
                    physics=True, hierarchical=False)
    return agraph(nodes=node_objs, edges=edge_objs, config=config)


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(layout="wide")
st.title("🚚 Algorithme Génétique pour 2E-EVRP")

with st.sidebar:
    st.header("⚙️ Paramètres")
    base_path = "./2E-EVRP-Instances-v2"
    type_choice = st.selectbox("Type :", ["Type_x", "Type_y"])
    customer_choice = st.selectbox("Nombre de clients :", [
                                   "Customer_5", "Customer_10", "Customer_15", "Customer_50", "Customer_100"])
    folder = os.path.join(base_path, type_choice, customer_choice)
    files = []
    if os.path.isdir(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".txt")]
    instance_choice = st.selectbox("Instance :", files)

    filepath = os.path.join(folder, instance_choice)
    nodes, L, C, Q, r, g, v = read_instance(filepath)

    vehicle_capacity = C
    battery_capacity = Q

    st.markdown("### 🚚 Capacités du véhicule")
    st.metric("Capacité véhicule (C)", f"{vehicle_capacity}")
    st.metric("Capacité batterie (Q)", f"{battery_capacity}")

    pop_size = st.slider("Population size", 10, 200, 50)
    max_gen = st.slider("Generations", 10, 500, 200)
    crossover_rate = st.slider("Crossover rate", 0.0, 1.0, 0.8)
    mutation_rate = st.slider("Mutation rate", 0.0, 1.0, 0.2)

if st.sidebar.button("🚀 Lancer GA") and instance_choice:
    try:
        nodes, L, C, Q, r, g, v = read_instance(filepath)
    except Exception as e:
        st.error(f"Erreur lecture instance: {e}")
        st.stop()

    vehicle_capacity = C
    battery_capacity = Q

    st.sidebar.markdown(
        f"**Params lus depuis instance:**  L={L}, C={C}, Q={Q}, r={r}, g={g}, v={v}")

    best_solution, best_score, best_info = GA_algorithm(
        nodes,
        vehicle_capacity,
        battery_capacity,
        pop_size=pop_size,
        max_gen=max_gen,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate
    )

    st.subheader(f"✅ Résultats pour {instance_choice}")
    st.write(f"**Score optimisé :** {best_score:.2f}")
    st.write(f"**Distance totale :** {best_info['distance']:.2f}")
    st.write(f"**Pénalités (coût) :** {best_info['penalty_cost']:.2f}")
    st.write(f"**Pénalités (retard) :** {best_info['delay_penalty']:.2f}")
    st.write(f"**Clients servis :** {best_info['clients_served']}")

    st.subheader("📊 Visualisation des tournées")
    plot = plot_solution_agraph(best_solution, nodes, best_info.get(
        'per_route_visits'), battery_capacity, vehicle_capacity)
    st.write(plot)

    st.write("🔎 Détails par tournée (visites avec % batterie après service) :")
    for i, visits in enumerate(best_info.get('per_route_visits', [])):
        if visits:
            st.write(f"Route {i+1}: " + " -> ".join(
                [f"{get_display_name(nodes[idx])}({pct:.1f}%)" for idx, pct, load in visits]))
        else:
            st.write(f"Route {i+1}: (vide)")
