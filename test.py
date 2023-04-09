import random  
import threading
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
radius = 3
backgroundColor = (255, 255, 255)
line_thickness = 5
line_color = (50, 100, 255)
city_color = (0, 0, 0)

#program variables
furthestDist = 0
cityPos = [{} for i in range(totalCities)]
cities = [[0 for j in range(totalCities)] for i in range(totalCities)]
cityDist = [[0 for j in range(totalCities)] for i in range(totalCities)]
cityRankings = [[0 for j in range(totalCities)] for i in range(totalCities)]
cityAverage = [0 for i in range(totalCities)]
distMean = [0]
path = [[0 for j in range(2)] for i in range(totalCities)]
pathToDraw = [[0 for j in range(2)] for i in range(totalCities)]


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

            pygame.draw.line(window, line_color, [cityPos[pathToDraw[i][0]]["x"], cityPos[pathToDraw[i][0]]["y"]], [cityPos[pathToDraw[i][1]]["x"], cityPos[pathToDraw[i][1]]["y"]], line_thickness)
            
            
        pygame.display.flip()
    pygame.quit()

generateCities()
model = torch.load("trained model")
fitness_function(model)

draw()
