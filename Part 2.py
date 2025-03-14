#ALL PAIRS - BELLMAN AND DIJKSTRA

#so the input given in the graph only (G)
#first, we check if the graph has:
#only positive weights -> dijkstra's algorithm
#contains negative weights as well -> bellman ford algorithm


'''Bellman Ford for all possible pairs:'''

#G.adj - key: node, value: nodes connected through edges
#G.weight - key: edge, value: weight of edge
#source - it's a list that takes the keys of G.adj, loops through each of the values
          #and if the value nodes are not in source already, adds them to.
          #to ensure that all nodes are included.
#loop through all source list
#at each iteration, consider each node a source
#perform nested loops on the source list and make a list of tuples called pairs:
#pairs is a list that stores all possible pairs of nodes.

#make a distance dictionary where
#key: pair of nodes, value: shortest distance btwn the pair of nodes.
#except for pairs (x, y) where x == y -> distance = 0
#set the distance = infinity

#perform the usual bellman ford algorithm on these pairs
#make note of the predecessor as well, with a predecessor dictionary
#where, the key: pair of nodes, value: predecessor node

#make sure the distance and predecessor dictionaries are global
#so we will loop through all possible sources,
#and will keep those values (from those source iterations) which gave the shortest distance for each pair.



def Bellman_all_pair(G):

    #get a list of all the nodes of the graph:
    nodes = []
    for k in G.adj:
        if k not in nodes:
            nodes.append(k)

        values = G.adj[k]
        for v in values:
            if v not in nodes:
                nodes.append(v)


    all_distances = {}
    all_predecessors = {}

    for src in nodes:
        all_distances[src] = {}
        all_predecessors[src] = {}

    print("empty all distances:", all_distances)
    print("empty all predecessors:", all_predecessors)



    #iterate through each node as source:
    for source in nodes:

        #VALUES OF all_distances and all_predecessors:
        #initialize, the distance and predecessor dictionaries:
        distance = {}
        predecessor = {}
        for u in nodes:
            for v in nodes:
                pair = (u,v)

                if u == v:
                    distance[pair] = 0

                else:
                    distance[pair] = math.inf

                predecessor[pair] = None
                
        
        for x in nodes:
            for y in nodes:

                #calculate the shortest distance from the source to each of the nodes:
                #all nodes are taken to be source vertices.
                #one source, per iteration.

                try: #not all edges exist acc. to G, since here, we're considering all possible pairs:
                    w = G.weights[(x,y)]
                    
                    if distance[(source, x)] + w < distance[(source, y)]:
                        distance[(source, y)] = distance[(source, x)] + w
                        predecessor[(source, y)] = x

                except KeyError: #when such edges do not exist
                    continue

        all_distances[source] = distance
        all_predecessors[source] = predecessor
                    


    return all_distances, all_predecessors


#-------------------------------------------------------------------------------------------------------------------------------------

def dijkstra(graph, source):
    # Initialize data structures
    shortest_distances = {node: float('inf') for node in graph.adj.keys()}
    predecessors = {node: None for node in graph.adj.keys()}
    shortest_distances[source] = 0

    # Priority queue (min heap) initialization
    pq = MinHeap([Node(value=source, key=0)])

    while not pq.is_empty():
        current_node = pq.extract_min()
        current = current_node.value

        for neighbor in graph.adjacent_nodes(current):
            weight = graph.w(current, neighbor)
            new_dist = shortest_distances[current] + weight
            if new_dist < shortest_distances[neighbor]:
                shortest_distances[neighbor] = new_dist
                predecessors[neighbor] = current
                pq.insert(Node(value=neighbor, key=new_dist))

    return shortest_distances, predecessors


def Dijkstra_all_pair(G):
    all_distances = {}
    all_predecessors = {}

    for source in G.adj.keys():
        all_distances[source], all_predecessors[source] = dijkstra(G, source)

    return all_distances, all_predecessors

    
        
                    
    
def all_pairs_shortest_path(G):

    contains_neg = False

    for np in G.weights:
        w = G.weights[np]
        
        if w < 0:
            contains_neg = True
            
    if contains_neg:
        
        dist, pred = Bellman_all_pair(G)
        print("distance:", dist)
        print("predecessor:", pred)
    else:
        return Dijkstra_all_pair(G)



print("\n\nALL PAIRS:")
print("\n1: bellman ford test case")
all_pairs_shortest_path(G)

print("\n2: dijkstra's test cases")
all_pairs_shortest_path(graph1)
all_pairs_shortest_path(graph)
