# TSP-NeuroHeuristics
TSP-NeuroHeuristics is a Python project that uses PyTorch and neuroevolution to train a neural network for solving the Traveling Salesman Problem. The network learns to generate heuristics that can approximate optimal solutions to the TSP

### Usage
To use this program, you'll need to have Python 3 and PyTorch installed on your system

To train the neural network, you can run the train.py script:

```
python train.py
```

By default, the program uses the following initial values:

``` python
#init values
totalCities = 10
radius = 3
backgroundColor = (200, 200, 200)   #gray
line_thickness = 1
line_color = (50, 100, 255)         #blue-ish
city_color = (0, 0, 0)              #black

populationSize = 20
generationSize = 100
mutationRate = 0.1
```

You can change these values by modifying the corresponding variables in the train.py script

### How It Works

#### (Step 1) City Generation:

The program randomly selects a group of cities and determines the distance between each pair of cities. It then assigns **three parameters** to each vertex that represents a path pointing away from a specific city.

If we consider *City A*, the **first parameter** is the average distance from *City A* to all other cities, which is added to the vertex.

The second parameter ranks all cities according to their distance from *City A*, and this ranking is added as the **second parameter** to the vertex that points from *City A* to the corresponding city.

The third parameter ranks *City A* based on its distance from the city it is pointing to, and this ranking is added as the **third parameter** to the vertex that points from *City A* to the corresponding city.

\* *The Averages are divided by the number of Cites to normalise them*

#### (Step 2) Neural Network Evaluation:

The program then takes the three parameters and inputs them into a neural network, which consists of an input layer with three nodes, four hidden layers with six nodes each, and an output layer with one node. The network evaluates weights or 'Heuristics' for each vertex based on these parameters, and this is done for all vertices.

<p>
    <img src="https://user-images.githubusercontent.com/98267072/230638500-887d8f37-3b31-4a05-ab4d-bf05a1693f05.png" width="200px"/>
    <br>
    <em>Neural Network Architecture</em>
</p>

Once all vertex weights are evaluated, the vertex with the highest weight is selected as the starting vertex. Then, the highest weighing vertex connected to that vertex is chosen as the next vertex in the path, and this process continues until the entire path from start to finish is generated.

<p>
    <img src="https://user-images.githubusercontent.com/98267072/230636319-793b1af3-e404-48b2-af1e-0817fc366d0c.gif" width="200px"/>
    <br>
    <em>Vertex selection/deletion visualisation</em>
</p>

\* *When a vertex is chosen from a city, all other vertecies from that city are deleted to make sure that that city won't get visited again*

#### (Step 3) Neuroevolution:
