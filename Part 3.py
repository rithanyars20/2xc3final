#A* ALGORITHM

def manhattan_distance(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return abs(x2 - x1) + abs(y2 - y1)


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



def A_Star(graph, start, goal,manhattan_distance):
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

        for neighbor in graph[current]:
            tentative_gScore = gScore[current] + graph[current][neighbor]
            if tentative_gScore < gScore.get(neighbor, float('inf')):
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + manhattan_distance(neighbor, goal)  # Use heuristic for neighbors
                openSet.push(neighbor, fScore[neighbor])

    return None, None  # No path found



# Example usage

# graph = {
#     'S': {'A': 4, 'B': 10, 'C': 11},
#     'A': {'B': 8, 'D': 5},
#     'B': {'D': 15},
#     'C': {'E': 20, 'F': 2, 'D': 8},
#     'D': {'H': 16, 'I': 20, 'F': 1},
#     'E': {'G': 19},
#     'F': {'G': 13},
#     'G': {},
#     'H': {'I': 1, 'J': 2},
#     'I': {'J': 5, 'K': 13},
#     'J': {'K': 7},
#     'K': {}
# }


# start = 'S'
# goal = 'G'
# heuristic = {'S': 11, 'A': 7, 'B': 8, 'C': 6, 'D': 6, 'E': 2, 'F': 4, 'G': 0, 'H': 6, 'I': 4, 'J': 4, 'K': 4}


# predecessors, shortest_path = A_Star(graph, start, goal, heuristic)
# print("Predecessors:", predecessors)
# print("Shortest path:", shortest_path)

coordinates_graph = {
    (0, 0): {(1, 0): 2, (0, 1): 4},
    (1, 0): {(1, 1): 5},
    (0, 1): {(1, 1): 1, (0, 2): 3},
    (1, 1): { (1, 2): 6},
    (0, 2): {(1, 2): 2}
}

start_node = (0, 0)
goal_node = (1, 2)

# Call the A* algorithm with the coordinates graph
cameFrom, path = A_Star(coordinates_graph, start_node, goal_node,manhattan_distance)


if path is not None:
    print("Path found:", path)
else:
    print("No path found")