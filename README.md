# TSP-NeuroHeuristics
TSP-NeuroHeuristics is an experimental Python project that uses PyTorch and neuroevolution to train a neural network for solving the Traveling Salesman Problem. The network learns to generate heuristics that can approximate optimal solutions to the TSP

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
backgroundColor = (255, 255, 255)   #white
line_thickness = 3
line_color = (50, 100, 255)         #blue-ish
city_color = (0, 0, 0)              #black

populationSize = 20
generationSize = 100
mutationRate = 0.1
```

You can change these values by modifying the corresponding variables in the train.py script

#### Training Visualisation
<p>
    <img src="https://user-images.githubusercontent.com/98267072/230741987-ac4ff1d8-ceac-42c8-ad9c-f375bc375acf.gif" width="250px"/>
    <br>
    <em>Cities: 10, Population: 20</em>
</p>

After the Model is done training it will be saved under "trained model". You can use the trained Model by running the test.py script:
```
python test.py
```

### Solved Examples
|100 Cities | 50 Cities | 25 Cities | 10 Cities|
|---|---|---|---|
|![100 city solved](https://user-images.githubusercontent.com/98267072/230771157-392dcf15-a9ca-48c3-a726-9ff205359ce3.png)|![50 city solved](https://user-images.githubusercontent.com/98267072/230771161-622421d1-6abb-4042-bc2b-655893d6978b.png)|![25 city solved](https://user-images.githubusercontent.com/98267072/230771160-a4275066-33fd-4703-bf4a-e7eb78845726.png)|![10 cities solved](https://user-images.githubusercontent.com/98267072/230771159-48f99014-bdd8-4f58-9189-bf7aef80a942.png)|

\* *They are not perfect but the heuristics used are pretty good. The model used to solve these was trained with the Prameters: Cities: 30, Population: 25, Generations: 1000*

### How It Works


#### (Step 1) City Generation:

The program randomly selects a group of cities and determines the distance between each pair of cities. It then assigns **three parameters** to each edge that represents a path pointing away from a specific city.

If we consider *City A*, the **first parameter** is the average distance from *City A* to all other cities, which is added to all edges pointing from *City A*.

The second parameter ranks all cities according to their distance from *City A*, and this ranking is added as the **second parameter** to the edge that points from *City A* to the corresponding city.

The third parameter ranks *City A* based on its distance from the city it is pointing to, and this ranking is added as the **third parameter** to the edge that points from *City A* to the corresponding city.

\* *The Averages are divided by the number of Cites to normalise them*

#### (Step 2) Neural Network Evaluation:

The program then takes the three parameters and inputs them into a neural network, which consists of an input layer with three nodes, four hidden layers with six nodes each, and an output layer with one node. The network evaluates weights or 'Heuristics' for each edge based on these parameters, and this is done for all edges.

<p>
    <img src="https://user-images.githubusercontent.com/98267072/230638500-887d8f37-3b31-4a05-ab4d-bf05a1693f05.png" width="200px"/>
    <br>
    <em>Neural Network Architecture</em>
</p>

Once all edge weights are evaluated, the edge with the highest weight is selected as the starting edge. Then, the highest weighing efge connected to that edge is chosen as the next edge in the path, and this process continues until the entire path from start to finish is generated.

<p>
    <img src="https://user-images.githubusercontent.com/98267072/230636319-793b1af3-e404-48b2-af1e-0817fc366d0c.gif" width="200px"/>
    <br>
    <em>Edge selection/deletion visualisation</em>
</p>

\* *When an Edge is chosen from a city, all other Edges from that city are deleted to make sure that that city won't get visited again*

#### (Step 3) Neuroevolution:
The genetic algorithm evolves a population of neural networks, where each network represents a potential solution to the TSP problem.

The population size is set to 20 in our case, which means that there are 20 different neural networks that are evolved simultaneously. The genetic algorithm evaluates the fitness of each network and selects the fittest ones to be used as parents for the next generation. The two parent networks are then combined using crossover and mutation operations to produce a new child network.

The crossover operation combines the weights of the parent networks to produce the weights of the child network. The mutation operation randomly changes some of the weights of the child network to introduce new variations. The resulting child network is then added to the population, and the process is repeated until a new generation of networks is produced.

This entire process is then repeated for a set number of generations, which is set to 100 in this code.

