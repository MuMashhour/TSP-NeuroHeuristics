import random  
import threading
import time
import math
import pygame
from pygame.locals import *
import numpy as np
import pprint
import torch
import os

class NeuralNet(torch.nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.layer1 = torch.nn.Linear(3, 6) # input layer to hidden layer 1
        self.layer2 = torch.nn.Linear(6, 6) # hidden layer 1 to hidden layer 2
        self.layer3 = torch.nn.Linear(6, 6) # hidden layer 2 to hidden layer 3
        self.layer4 = torch.nn.Linear(6, 6) # hidden layer 3 to hidden layer 4
        self.layer5 = torch.nn.Linear(6, 1) # hidden layer 4 to output layer

    def forward(self, x):
        x = torch.relu(self.layer1(x)) # apply ReLU activation to hidden layer 1
        x = torch.relu(self.layer2(x)) # apply ReLU activation to hidden layer 2
        x = torch.relu(self.layer3(x)) # apply ReLU activation to hidden layer 3
        x = torch.relu(self.layer4(x)) # apply ReLU activation to hidden layer 4
        x = self.layer5(x) # no activation on output layer
        return x


#init values
totalCities = 10
cityPos = [{} for i in range(totalCities)]
radius = 3
backgroundColor = (200, 200, 200)
line_thickness = 1
line_color = (50, 100, 255)
city_color = (0, 0, 0)

populationSize = 20
generationSize = 100
mutationRate = 0.1

#program variables
furthestDist = 0
cities = [[0 for j in range(totalCities)] for i in range(totalCities)]
cityDist = [[0 for j in range(totalCities)] for i in range(totalCities)]
cityRankings = [[0 for j in range(totalCities)] for i in range(totalCities)]
cityAverage = [0 for i in range(totalCities)]
distMean = [0]
path = [[0 for j in range(2)] for i in range(totalCities)]
pathToDraw = [[0 for j in range(2)] for i in range(totalCities)]

#generate cities and weights from NN
def generateCities():
    #generate city Positions 0-600
    for i in range(totalCities):
        cityPos[i] = {"x": random.randrange(0,600), "y": random.randrange(0,600)}

    #get NN input Values
    for i in range(totalCities):
        for j in range(totalCities):
            cityRankings[i][j] = (math.dist([cityPos[i]["x"], cityPos[i]["y"]], [cityPos[j]["x"], cityPos[j]["y"]]))
            cityAverage[i] += cityRankings[i][j]
    
        cityDist[i] = cityRankings[i]
        furthestDist = max(cityDist[i])
        cityAverage[i] /= (totalCities * furthestDist)

        cityRankings[i] = sorted(cityRankings[i])
    for i in range(totalCities):
        for j in range(totalCities):
            cities[i][j] = {"average": cityAverage[i], "Ranking": (cityRankings[i].index(cityDist[i][j]) / totalCities), "Ranking2": (cityRankings[j].index(cityDist[i][j])) / totalCities}
   
    distMean[0] = np.mean(cityDist)
generateCities()

#draw function
def draw():
    #print(path)
    pygame.init()
    window = pygame.display.set_mode((600, 600))
    window.fill(backgroundColor)

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == QUIT:
                run = False

        
        window.fill(backgroundColor)

        myfont = pygame.font.SysFont("monospace", 20)
        myfont2 = pygame.font.SysFont("monospace", 15)

        for i in range(totalCities):  
            cityID = myfont.render(str(i + 1), 1, city_color)
            cityCOO = myfont2.render("(" + str(cityPos[i]["x"]) + ", " + str(cityPos[i]["y"]) + ")", 1, city_color)

            window.blit(cityID, [cityPos[i]["x"], cityPos[i]["y"]])
            #window.blit(cityCOO, [cityPos[i]["x"] - 10, cityPos[i]["y"] + 20])

            pygame.draw.circle(window, city_color, [cityPos[i]["x"], cityPos[i]["y"]], radius)

            for j in range(line_thickness):
                pygame.draw.line(window, line_color, [cityPos[pathToDraw[i][0]]["x"] + j, cityPos[pathToDraw[i][0]]["y"] + j], [cityPos[pathToDraw[i][1]]["x"] + j, cityPos[pathToDraw[i][1]]["y"] + j])
            
            
        pygame.display.flip()
    pygame.quit()

def learn():
    fittestDist = [0]
    def fitness_function(net):
        vertices = [[-999999999 for j in range(totalCities)] for i in range(totalCities)]
        #vertex weight from neural net
        for i in range(totalCities):
            for j in range(totalCities):  
                if(i == j):
                    continue

                input_data = torch.tensor(list(cities[i][j].values())) 
                output = net(input_data) # pass the input through the network to get an output

                if(vertices[i][j] == -999999999):
                    vertices[i][j] = 1;
                    vertices[j][i] = 1;

                vertices[i][j] *= output.item()
                vertices[j][i] *= output.item()
                #print(output, cityDist[i][j], nodes[i][j])

        #path based on weights
        pathIndex = 0;
        verticesMod = vertices
        start = max(max(verticesMod))
        startNode = 0
        gotStartNode = False
        dist = 0;

        #get start node
        if(gotStartNode == False):
            for i in range(totalCities):
                try:
                    pos = [i, verticesMod[i].index(start)]
                    startNode = i
                    gotStartNode = True
                    break
                except:
                    continue
    
        path[pathIndex] = pos
        pathIndex += 1
        dist += cityDist[pos[0]][pos[1]]

        for j in range(totalCities):
            verticesMod[pos[0]][j] = -999999999
            verticesMod[j][pos[0]] = -999999999
        for i in range(totalCities - 1):
        

            if(i == totalCities - 2):
                pos = [pos[1], startNode]
            else:
                pos = [pos[1], verticesMod[pos[1]].index(max(verticesMod[pos[1]]))]   
                #pprint.pprint(verticesMod)


            for j in range(totalCities):
                verticesMod[pos[0]][j] = -999999999
                verticesMod[j][pos[0]] = -999999999

            path[pathIndex] = pos
            pathIndex += 1
            dist += cityDist[pos[0]][pos[1]]
        
    
        if(fittestDist[0] > dist or fittestDist[0] == 0):
            fittestDist[0] = dist

            for i in range(totalCities):
                for j in range(2):
                    pathToDraw[i][j] = path[i][j]

        return dist/distMean[0]
    # Define the neuroevolution algorithm
    class NeuroEvolution():
        def __init__(self, population_size, mutation_rate):
            self.population_size = population_size
            self.mutation_rate = mutation_rate
            self.population = []
            self.fitness_scores = []

            for i in range(population_size):
                self.population.append(NeuralNet())

        def select_parents(self):
            # Use tournament selection to choose the fittest individuals as parents
            parent1_idx = np.random.randint(self.population_size)
            parent2_idx = np.random.randint(self.population_size)

            while parent2_idx == parent1_idx:
                parent2_idx = np.random.randint(self.population_size)

            if self.fitness_scores[parent1_idx] < self.fitness_scores[parent2_idx]:
                return parent1_idx
            else:
                return parent2_idx

        def mutate(self, net):
            # Randomly mutate a weight of the neural network with the specified mutation rate
            for param in net.parameters():
                if np.random.rand() < self.mutation_rate:
                    param.data += torch.randn_like(param)

        def evolve(self, fitness_function, num_generations):
            for gen in range(num_generations):
                # Evaluate the fitness of each individual in the population
                self.fitness_scores = []
                for i in range(self.population_size):
                    fitness_score = fitness_function(self.population[i])
                    self.fitness_scores.append(fitness_score)

                # Select the fittest individuals as parents and use them to create the next generation
                new_population = []
                for i in range(self.population_size):
                    parent1_idx = self.select_parents()
                    parent2_idx = self.select_parents()
                    parent1 = self.population[parent1_idx]
                    parent2 = self.population[parent2_idx]
                    child = NeuralNet()
                    for param_child, param_parent1, param_parent2 in zip(child.parameters(), parent1.parameters(), parent2.parameters()):
                        # Crossover: combine the weights of the two parents randomly
                        param_child.data = torch.where(torch.rand_like(param_child) < 0.5, param_parent1.data, param_parent2.data)
                    # Mutate the child before adding it to the new population
                    self.mutate(child)
                    new_population.append(child)

                # Replace the old population with the new population
                self.population = new_population

                # Print the fittest individual in the population for this generation
                print(f"Generation {gen+1}: Best fitness = {min(self.fitness_scores)}")
                generateCities()
                fittestDist[0] = 0

            # Return the fittest individual in the final population
            return self.population[np.argmin(self.fitness_scores)]
    # Create an instance of the NeuroEvolution class with a population size of 10 and a mutation rate of 0.1
    ne = NeuroEvolution(population_size=populationSize, mutation_rate=mutationRate)
    # Evolve the population for 100 generations using the fitness function defined above
    fittest_net = ne.evolve(fitness_function, num_generations=generationSize)

    weights = fittest_net.state_dict()

    print("-------------------------------------------------------------------- \n")
    print("fittest Model: \n")

    pprint.pprint(weights)

t1 = threading.Thread(target=learn)
t1.start()

draw()

t1.join()
