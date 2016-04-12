from functools import total_ordering
from heapq import nlargest
from math import fabs
from itertools import dropwhile
from operator import itemgetter, add, sub, mul, truediv, pow, xor
from random import getrandbits, randint, randrange, sample
from sys import maxsize


def split_str(string: str, n: int):
    while string:
        yield string[:n]
        string = string[n:]


def get_rand_chromosome(bits: int):
    return int(getrandbits(bits) + (1 << 32), 2)[2:]


class Genotype:
    def __init__(self, chromosome_length, mutation_rate, crossover_rate, kill_rate, max_population):
        self.chromosome_length = 32
        self.mutation_rate = 0.001
        self.crossover_rate = 0.7
        self.kill_rate = 0.3
        self.max_population = 2000


class Tournament:
    tourney_size = 10
    num_victors = 4


@total_ordering
class Organism:
    from .graycode import graycode
    gene_pool = dict(zip(graycode(4), [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, add,
        sub, mul, truediv, pow, xor]))

    def __init__(self, genotype: Genotype):
        self.bits = get_rand_chromosome(self.genotype.chromosome_length)
        self.genotype = genotype

    def __float__(self):
        return float(self._eval())

    def __len__(self):
        return len(self.bits)

    def __getitem__(self, item):
        return self.bits[item]

    def __iter__(self):
        return iter(self.bits)

    def __setslice__(self, i, j, sequence):
        return self.bits[i:j:sequence]

    def __str__(self):
        return self.bits

    def __repr__(self):
        return self.bits

    def __hash__(self):
        return hash(self.bits)

    def __le__(self, other):
        return float(self) <= float(other)

    def pprint(self):
        return

    @staticmethod
    def crossover(first, second):
        cut_pos = randint(0, len(min(first, second))-1)
        ls = [first[:cut_pos], second[:cut_pos]]
        rs = [second[cut_pos:], first[cut_pos:]]

        org_one, org_two = Organism(first.genotype), Organism(first.genotype)
        org_one.bits = ls[0] + rs[0]
        org_two.bits = ls[1] + rs[1]

        return [org_one, org_two]

    def _decode(self):
        ops = []
        for gene in split_str(self.bits, 4):
            try:
                ops.append(self.gene_pool[gene])
            except KeyError:
                pass
        return ops

    def _eval(self):
        genes = self._decode()
        last_op = None
        genes = list(dropwhile(lambda g: not isinstance(g, int), genes))

        if len(genes) == 0:
            return 1000.0

        val = genes.pop(0)
        for gene in genes:
            is_dec = isinstance(gene, int)
            if last_op is None and is_dec:
                pass
            elif last_op is None and not is_dec:
                last_op = gene
            elif last_op is not None and is_dec:
                try:
                    val = last_op(val, gene)
                except ZeroDivisionError:
                    pass
                last_op = None
            elif last_op is not None and not is_dec:
                pass

        return val

    def fitness(self, base: float):
        try:
            v = 1.0/(fabs(base - float(self)))
            return v
        except ZeroDivisionError:
            return maxsize

    def mutate(self):
        idx = randint(0, len(self.bits)-1)
        self.bits = str(bit if x != idx else str(~bool(bit)) for bit, x in zip(self.bits, iter(int, 1)))

    def matewith(self, other):
        chldrn = []
        pr = 1 / randint(1, 10)
        if pr <= 0.4:
            num_children = 2
        elif pr <= 0.6:
            num_children = 1
        else:
            num_children = 0
        for _ in range(num_children):
            pr = 1 / randint(1, 10)
            if pr <= Organism.crossover_rate:
                children.extend(Organism.crossover(self, other))
            elif pr <= Organism.crossover_rate + 0.2:
                org = Organism()
                org.bits = self.bits
                chldrn.append(org)

        return chldrn


def half_pairs(l):
    lngth = len(l) // 2
    return zip(l[:lngth], l[lngth:])


def randlist(start, end, num):
    return (randrange(start, end) for _ in range(num))

if __name__ == "__main__":
    set_point = 19.57
    organisms = [Organism() for _ in range(1000)]

    winners = set()

    for tick in range(20):
        total_num_participants = Organism.tourney_size * Organism.num_tourneys
        participant_pool = sample(organisms, total_num_participants)
        organisms = organisms[total_num_participants:]

        print('Max number of participants', total_num_participants)
        print('Pool Size', len(participant_pool))
        print('Organism count', len(organisms))

        fitnesses = dict()
        for participant in participant_pool:
            fitnesses[participant] = participant.fitness(set_point)

        victors = list()
        losers = list()
        for tourney_num in range(Organism.num_tourneys):
            size = Organism.tourney_size
            participants = itemgetter(*randlist(0, size-1, size))(participant_pool)
            winners = nlargest(Organism.num_victors, participants)
            total_fitness = sum(participants)

        print('Fitness', total_fitness)

        children = []
        print('Victor Count', len(victors))
        for (vic_one, vic_two) in half_pairs(victors):
            alpha, beta = (vic_one, vic_two) if vic_one >= vic_two else (vic_two, vic_one)
            children.extend(alpha.matewith(beta))

        print('Children count', len(children))

        mut_count = 0
        for organism in organisms:
            roll = randint(1, 1000)
            if roll <= 4:
                mut_count += 1
                organism.mutate()
            elif roll <= 8:
                del organisms[organisms.index(organism)]

        print('Mut count', mut_count)
        print('Organism count (pre-kill)', len(organisms))
        organisms = [x for x in organisms if x.fitness(set_point) >= x.kill_rate]
        print('Organism count (pre-birth)', len(organisms))
        organisms.extend(children)
        print('Organism count (post-kill)', len(organisms), "\n")
        print('')

    print([(x, float(x), x.fitness(set_point)) for x in reversed(sorted(organisms, key=lambda x: x.fitness(set_point)))])
