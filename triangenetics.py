from heapq import heappush, heappop, heapify
import numpy as np
from PIL import Image, ImageDraw, ImageMath
from random import seed, gauss, randrange, random
from sys import maxsize

seed(2)

reference = 'sonic.png'
im = Image.open(reference, 'r')
size = im.size


class Gene:
    """
    Each organism has multiple genes. Genes are what are mutated and modified.
    This gene contains a radius r, a rotation t and a color c
    """
    def __init__(self, *args, **kwargs):
        self.params = dict()
        for k, v in self.mutables.items():
            try:
                left, right = self.bounds[k]
            except KeyError:
                left, right = None, None
            if v == float:  # initialize to random, [0, 1]
                val = random() if not left else randrange(left, right)
                self.__setattr__(k, v(val))
            elif v == int:
                start, end = (0, 100) if not left else (left, right)
                self.__setattr__(k, v(randrange(start, end)))
            elif v == Position:
                xbnd, ybnd = size if not left else (left, right)
                val = v(randrange(0, xbnd), randrange(0, ybnd))
                self.__setattr__(k, val)
            else:
                self.__setattr__(k, v())

    def __str__(self):
        return str(self.params)

    def __repr__(self):
        return str(self)


class Color:
    """
    A color consists of four floating point values, one for each of RGBA
    """
    def __init__(self, r=0, g=0, b=0, a=1.0):
        self.rgba = (r, g, b, a)

        if r == g == b == 0 and a == 1.0:
            self.randomize()

    def randomize(self):
        self.rgba = (randrange(0, 255),
                     randrange(0, 255),
                     randrange(0, 255),
                     randrange(0, 255))

    def __str__(self):
        return "({}, {}, {}, {})".format(*self.rgba)

    def __repr__(self):
        return str(self)


class Position:
    def __init__(self, xpos=0.0, ypos=0.0):
        self.position = (xpos, ypos)

        if xpos == 0.0 and ypos == 0.0:
            self.randomize()

    def randomize(self):
        self.position = (random(), random())

    def __str__(self):
        return "({}, {})".format(*self.position)

    def __repr__(self):
        return str(self)


class TriangleGene(Gene):
    mutables = {'radius': float,
                'color': Color,
                'position': Position}

    bounds = {'position': size, 'radius': (30, 70)}

    @property
    def verts(self):
        x, y = self.position.position
        rad = self.radius
        return [(x - rad // 2, y), (x, y + rad), (x + rad // 2, y)]

    def __init__(self):
        super().__init__()


class Organism:
    initial_population_size = 200
    max_population = 600
    number_of_genes = 100
    mutation_rate = 0.01
    crossover_rate = 0.7
    kill_rate = 0.3
    gene_split_rate = 0.3

    # Apply k-means clustering over two organisms to
    # find similar high density fitness areas
    # fitness chunks done in groups
    # variable length genes

    mutables = {'chromosome_length': float,
                'mutation_rate': float,
                'crossover_rate': float,
                'kill_rate': float}

    def __init__(self):
        self.genes = list(TriangleGene() for _ in range(self.number_of_genes))
        self.fitness = 0

    @staticmethod
    def fitness_of(arr):
        sub = np.subtract(im, arr)
        try:
            return 1/np.mean(sub)
        except ZeroDivisionError:
            return maxsize

    def crossover(self, other):
        pass


organisms = list(Organism() for _ in range(Organism.initial_population_size))


def draw_org(organism):
    img = Image.new("RGBA", size)
    draw = ImageDraw.Draw(img)
    for gene in organism.genes:
        draw.polygon(gene.verts, gene.color.rgba)

    return img


def draw_organisms(population, fn):
    total_fitness = 0
    for o in population:
        img = draw_org(o)
        fitness = o.fitness_of(img)
        total_fitness += fitness
        fn(id(o), fitness)

    print(total_fitness, total_fitness / len(population))


draw_organisms(organisms, print)
