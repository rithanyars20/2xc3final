#COMPARING PERFORMANCE OF DIJKSTRA's AND A* ALGORITHM:

import math
import csv
import timeit
import matplotlib.pyplot as plt

class WeightedGraph:

    def __init__(self,nodes):
        self.graph=[]
        self.weights={}
        for node in range(nodes):
            self.graph.append([])

    def add_node(self,node):
        self.graph[node]=[]

    def add_edge(self, node1, node2, weight):
        if node2 not in self.graph[node1]:
            self.graph[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def get_weights(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def are_connected(self, node1, node2):
        for neighbour in self.graph[node1]:
            if neighbour == node2:
                return True
        return False

    def get_neighbors(self, node):
        return self.graph[node]

    def get_number_of_nodes(self,):
        return len(self.graph)
    
    def get_nodes(self,):
        return [i for i in range(len(self.graph))]

class DirectedWeightedGraph:

    def __init__(self):
        self.adj = {}
        self.weights = {}

    def are_connected(self, node1, node2):
        for neighbour in self.adj[node1]:
            if neighbour == node2:
                return True
        return False

    def adjacent_nodes(self, node):
        return self.adj[node]

    def add_node(self, node):
        self.adj[node] = []

    def add_edge(self, node1, node2, weight):
        if node2 not in self.adj[node1]:
            self.adj[node1].append(node2)
        self.weights[(node1, node2)] = weight

    def w(self, node1, node2):
        if self.are_connected(node1, node2):
            return self.weights[(node1, node2)]

    def number_of_nodes(self):
        return len(self.adj)

def init_d(G):
    n = G.number_of_nodes()
    d = [[float("inf") for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(n):
            if G.are_connected(i, j):
                d[i][j] = G.w(i, j)
        d[i][i] = 0
    return d

#1.1

class Node:
    def __init__(self, value, key):
        self.value = value
        self.key = key

    def __str__(self):
        return f"({self.value}, {self.key})"

class MinHeap:
    def __init__(self, data):
        self.items = data
        self.length = len(data)
        self.build_heap()

        self.map = {}
        for i in range(self.length):
            self.map[self.items[i].value] = i

    def find_left_index(self, index):
        return 2 * (index + 1) - 1

    def find_right_index(self, index):
        return 2 * (index + 1)

    def find_parent_index(self, index):
        return (index + 1) // 2 - 1

    def sink_down(self, index):
        smallest_known_index = index

        left_index = self.find_left_index(index)
        right_index = self.find_right_index(index)

        if left_index < self.length and self.items[left_index].key < self.items[index].key:
            smallest_known_index = left_index

        if right_index < self.length and self.items[right_index].key < self.items[smallest_known_index].key:
            smallest_known_index = right_index

        if smallest_known_index != index:
            self.items[index], self.items[smallest_known_index] = self.items[smallest_known_index], self.items[index]
            self.map[self.items[index].value] = index
            self.map[self.items[smallest_known_index].value] = smallest_known_index
            self.sink_down(smallest_known_index)

    def build_heap(self):
        for i in range(self.length // 2 - 1, -1, -1):
            self.sink_down(i)

    def insert(self, node):
        if len(self.items) == self.length:
            self.items.append(node)
        else:
            self.items[self.length] = node
        self.map[node.value] = self.length
        self.length += 1
        self.swim_up(self.length - 1)

    def swim_up(self, index):
        while index > 0 and self.items[self.find_parent_index(index)].key > self.items[index].key:
            self.items[index], self.items[self.find_parent_index(index)] = self.items[self.find_parent_index(index)], self.items[index]
            self.map[self.items[index].value] = index
            self.map[self.items[self.find_parent_index(index)].value] = self.find_parent_index(index)
            index = self.find_parent_index(index)

    def extract_min(self):
        self.items[0], self.items[self.length - 1] = self.items[self.length - 1], self.items[0]
        self.map[self.items[self.length - 1].value] = self.length - 1
        self.map[self.items[0].value] = 0

        min_node = self.items[self.length - 1]
        self.length -= 1
        self.sink_down(0)
        return min_node

    def is_empty(self):
        return self.length == 0

def dijkstra(graph, source, k):
    distance = {node: math.inf for node in graph.adj.keys()}
    distance[source] = 0
    path = {node: [] for node in graph.adj.keys()}

    pq = MinHeap([Node(value=source, key=0)])
    relax_count = {node: 0 for node in graph.adj.keys()}

    while not pq.is_empty():
        current_node = pq.extract_min()
        current = current_node.value

        if relax_count[current] <= k:
            for neighbor in graph.adjacent_nodes(current):
                weight = graph.w(current, neighbor)
                new_dist = distance[current] + weight
                if new_dist < distance[neighbor]:
                    distance[neighbor] = new_dist
                    pq.insert(Node(value=neighbor, key=new_dist))
                    path[neighbor] = path[current] + [current]
                    relax_count[neighbor] += 1

    return distance, path


def convert_edges_to_coordinates_graph(edges):
    coordinates_graph = {}
    for edge in edges:
        node1, node2, weight = edge
        if node1 not in coordinates_graph:
            coordinates_graph[node1] = {}
        if node2 not in coordinates_graph:
            coordinates_graph[node2] = {}
        coordinates_graph[node1][node2] = weight
        coordinates_graph[node2][node1] = weight
    return coordinates_graph

def manhattan_distance(node1, node2):
    return abs(node2 - node1)

class PriorityQueue:
    def __init__(self):
        self.queue = []

    def push(self, item, priority):
        self.queue.append((priority, item))
        self.queue.sort()

    def pop(self):
        if not self.is_empty():
            return self.queue.pop(0)[1]
        else:
            raise IndexError("pop from empty priority queue")

    def remove(self, item):
        self.queue = [(priority, value) for priority, value in self.queue if value != item]

    def is_empty(self):
        return len(self.queue) == 0

def reconstruct_path(cameFrom, current):
    path = []
    while current in cameFrom:
        path.append(current)
        current = cameFrom[current]
    path.append(current)
    path.reverse()
    return path

def A_Star(graph, start, goal, manhattan_distance):
    openSet = PriorityQueue()
    openSet.push(start, 0)
    cameFrom = {}
    gScore = {node: float('inf') for node in graph}
    gScore[start] = 0
    fScore = {node: float('inf') for node in graph}
    fScore[start] = manhattan_distance(start, goal)  # Use the heuristic for the start node

    while not openSet.is_empty():
        current = openSet.pop()

        if current == goal:
            return (cameFrom, reconstruct_path(cameFrom, current))

        if current not in graph:
            continue

        for neighbor in graph[current]:
            tentative_gScore = gScore[current] + graph[current][neighbor]
            if tentative_gScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + manhattan_distance(neighbor, goal)  # Use heuristic for neighbors
                openSet.push(neighbor, fScore[neighbor])

    return None, None  # No path found

def part4(sourcestation, destinationstation, totalstations):
    #Heuristic function
    g = WeightedGraph(totalstations)
    g2 = DirectedWeightedGraph()
    with open('london_stations.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            #print(int(row['id']))
            g.add_node(int(row['id']))
            g2.add_node(int(row['id']))
    #print("nodes: ",g.get_nodes())
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    nestedlist = []
    with open('london_connections.csv', newline='') as london_connections:
            london_connectionsread = csv.DictReader(london_connections)
            for row in london_connectionsread:
                #print("london_connections rows: ",row['station1'], row['station2'])
                with open('london_stations.csv', newline='') as csvfile:
                    reader = csv.DictReader(csvfile)
                    readercopy = csv.DictReader(csvfile)
                    for row2 in reader:
                        #print("london_staions1 rows: ",row2['id'])
                        if row['station1'] == row2['id']:
                            x1 = float(row2['latitude'])
                            y1 = float(row2['longitude'])
                            #print(x1, y1)
                            break
                    for row3 in reader:
                        #print("london_staions2 rows: ",row3['id'])
                        if row['station2'] == row3['id']:
                            x2 = float(row3['latitude'])
                            y2 = float(row3['longitude'])
                            #print(x2, y2)
                            break
                    g.add_edge(int(row['station1']), int(row['station2']), math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                    g2.add_edge(int(row['station1']), int(row['station2']), math.sqrt((x2 - x1)**2 + (y2 - y1)**2))
                    nestedlist.append([int(row['station1']), int(row['station2']), math.sqrt((x2 - x1)**2 + (y2 - y1)**2)])
    # print("graph: ",g.graph)
    # print(g.get_weights(sourcestation,destinationstation))
    # print("nestedlist: ",nestedlist)
    #Heuristic function

    #A*
    graph = convert_edges_to_coordinates_graph(nestedlist)

    start_node = sourcestation
    goal_node = destinationstation
    Astartimestart = timeit.default_timer()
    cameFrom, path = A_Star(graph, start_node, goal_node, manhattan_distance)
    Astartimestop = timeit.default_timer()
    Astartime = Astartimestop - Astartimestart
    print("Came From:", cameFrom)
    print("Path:", path)
    #A*

    #Dijkstra
    source_node = sourcestation

    # Relaxation limit
    k = 2

    # Run Dijkstra's algorithm
    Dijkstratimestart = timeit.default_timer()
    shortest_distances, shortest_paths = dijkstra(g2, source_node, k)
    Dijkstratimestop = timeit.default_timer()
    Dijkstratime = Dijkstratimestop - Dijkstratimestart
    # Print the shortest distances
    print("Shortest distances from node", shortest_distances)
    # Print the shortest paths
    print("\nShortest paths from node", shortest_paths)
        #Dijkstra

    print("A* time: ", Astartime)
    print("Dijkstra time: ", Dijkstratime)

    x_values = []
    for i in range(501):
        x_values.append(i)

    print("x_values: ",len(x_values))
    y1 = []
    y2 = []
    for i in range(501):
        Astartimestart = timeit.default_timer()
        cameFrom, path = A_Star(graph, start_node, goal_node, manhattan_distance)
        Astartimestop = timeit.default_timer()
        Astartime = Astartimestop - Astartimestart
        y1.append(Astartime)
        Dijkstratimestart = timeit.default_timer()
        shortest_distances, shortest_paths = dijkstra(g2, source_node, k)
        Dijkstratimestop = timeit.default_timer()
        Dijkstratime = Dijkstratimestop - Dijkstratimestart
        y2.append(Dijkstratime)
    print("y1: ",y1)
    print("y2: ",y2)
    # Use plt.scatter to plot individual points
    plt.plot(x_values, y1, label='A*', color='r')
    plt.plot(x_values, y2, label='Djikstra', color='g')

    plt.xlabel("Number of iterations")
    plt.ylabel("Runtimes")
    plt.title("Comparison of Shortest Path Algorithms")
    plt.legend()
    plt.show()

#Test Cases
print("Djikstra is faster than A* for indirect connections (same lines): ")
part4(49, 279, 304) #Djikstra is faster than A* for indirect connections (same lines)

print("A* outperforms Djikstra for direct connections (same lines):")
part4(11, 163, 304) #A* outperforms Djikstra for direct connections (same lines)

print("Djikstra is much faster than A* for several transfers:")
part4(267, 180, 304) #Djikstra is much faster than A* for several transfers

print("Adjacent lines (comparable):")
part4(14, 90, 304) #Adjacent lines (comparable)