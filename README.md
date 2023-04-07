# TSP-NeuroHeuristics
TSP-NeuroHeuristics is a Python project that uses PyTorch and neuroevolution to train a neural network for solving the Traveling Salesman Problem. The network learns to generate heuristics that can approximate optimal solutions to the TSP

### How It Works

The program randomly selects a group of cities and determines the distance between each pair of cities. It then assigns **three parameters** to each vertex that represents a path pointing away from a specific city.

If we consider *City A*, the **first parameter** is the average distance from *City A* to all other cities, which is added to the vertex.

The second parameter ranks all cities according to their distance from *City A*, and this ranking is added as the **second parameter** to the vertex that points from *City A* to the corresponding city.

The third parameter ranks *City A* based on its distance from the city it is pointing to, and this ranking is added as the **third parameter** to the vertex that points from *City A* to the corresponding city.

\* *The Averages are divided by the number of Cites to normalise them*

The program then takes the three parameters and inputs them into a neural network, which consists of an input layer with three nodes, four hidden layers with six nodes each, and an output layer with one node. The network evaluates weights for each vertex based on these parameters, and this is done for all vertices.

Once all vertex weights are evaluated, the vertex with the highest weight is selected as the starting vertex. Then, the highest weighing vertex connected to that vertex is chosen as the next vertex in the path, and this process continues until the entire path from start to finish is generated.

\* *When a vertex is chosen from a city we make sure to delete all other vertecies from that city to make sure we don't visit that city again*
