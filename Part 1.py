#DIJKSTRA AND BELLMAN FORD

#DIJKSTRA

import random
import math

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


# Test Case 1: Simple graph with no relaxation needed
print("First test case:")
graph1 = DirectedWeightedGraph()
graph1.add_node(0)
graph1.add_node(1)
graph1.add_node(2)
graph1.add_node(4)

graph1.add_edge(0, 1, 2)
graph1.add_edge(0, 2, 4)
graph1.add_edge(1, 2, 1)

source_node1 = 0
k1 = 3

shortest_distances1, shortest_paths1 = dijkstra(graph1, source_node1, k1)
print("Test Case 1:")
print("Shortest distances:", shortest_distances1)
print("Shortest paths:", shortest_paths1)

print("\nSecond test case:")
# Create a weighted graph instance
graph = DirectedWeightedGraph()

# Add edges with weights
graph.add_node(0)
graph.add_node(1)
graph.add_node(2)
graph.add_node(3)
graph.add_node(4)
graph.add_node(5)
graph.add_node(6)
graph.add_node(7)

graph.add_edge(0, 1, 3)
graph.add_edge(0, 2, 1)
graph.add_edge(1, 3, 2)
graph.add_edge(2, 1, 4)
graph.add_edge(2, 4, 2)
graph.add_edge(3, 5, 3)
graph.add_edge(3, 6, 4)
graph.add_edge(4, 3, 1)
graph.add_edge(4, 5, 2)
graph.add_edge(5, 7, 3)
graph.add_edge(6, 7, 2)

# Source node
source_node = 0

# Relaxation limit
k = 2

# Run Dijkstra's algorithm
shortest_distances, shortest_paths = dijkstra(graph, source_node, k)

# Print the shortest distances
print("Shortest distances from node", shortest_distances)
# Print the shortest paths
print("\nShortest paths from node", shortest_paths)


#1.2


#BELLMAN FORD
import random
import math

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


'''
Bellman Ford's algorithm:

directed graphs
self.adj = {node: [nodes it reaches]}

key - nodes from which the edges start
value - list of nodes from which the edges end when started from key node.

'''

def BellmanFord (G, source, k):

    distance = {} #key: node,  value: shortest distance
    path = {} #key: node,  value: path node

    for vertex in G.adj:
        distance[vertex] = math.inf
        path[vertex] = None

    #set distance to source as 0
    distance[source] = 0


    for _ in range(k):
        for u in distance:
            for v in G.adj[u]:
                w = G.weights[u,v]
                if distance[u] + w < distance[v]:
                    distance[v] = distance[u] + w
                    path[v] = u
    
    return distance, path
    
G = DirectedWeightedGraph()  # Instantiate the DirectedWeightedGraph class

G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node(4)

G.add_edge(1, 2, 4)
G.add_edge(1, 4, 5)
G.add_edge(4, 3, 3)
G.add_edge(3, 2, -10)

d, p = BellmanFord(G, 1, 3)  # Call the BellmanFord function with the instantiated graph
print("distance:", d)
print("path:", p)





#1.3

