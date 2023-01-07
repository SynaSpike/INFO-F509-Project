

# code from chapGPT

import random

class PublicGoodGame:
    def init(self, population_size, wealth_ratio, group_size, initial_endowment, threshold, mutation_prob):
        self.population_size = population_size
        self.wealth_ratio = wealth_ratio
        self.group_size = group_size
        self.initial_endowment = initial_endowment
        self.threshold = threshold
        self.mutation_prob = mutation_prob

        self.population = []
        self.create_population()

    def create_population(self):
        num_rich = int(self.population_size * self.wealth_ratio)
        num_poor = self.population_size - num_rich

        for i in range(num_rich):
          self.population.append(Individual(True, self.initial_endowment[0]))
        for i in range(num_poor):
          self.population.append(Individual(False, self.initial_endowment[1]))

    def play_game(self):
        groups = self.create_groups()
        for group in groups:
            contributions = []
            for individual in group:
                if individual.strategy == "cooperate":
                    contribution = individual.endowment * self.contribution_fraction
                    individual.endowment -= contribution
                    contributions.append(contribution)
                else:
                    contributions.append(0)

            total_contribution = sum(contributions)
            if total_contribution >= self.threshold:
                for i, individual in enumerate(group):
                    individual.endowment += contributions[i]
            else:
                for individual in group:
                    individual.endowment = 0

    def create_groups(self):
        random.shuffle(self.population)
        groups = [self.population[i:i+self.group_size] for i in range(0, len(self.population), self.group_size)]
        return groups

    def update_strategies(self):
        for individual in self.population:
            other = random.choice(self.population)
            prob = self.fermi_function(individual.fitness, other.fitness)
            if random.uniform(0, 1) < prob:
                individual.strategy = other.strategy
            if random.uniform(0, 1) < self.mutation_prob:
                individual.strategy = "cooperate" if individual.strategy == "defect" else "defect"

    def fermi_function(self, x, y):
        return 1 / (1 + math.exp(-self.k * (x - y)))

    def calculate_fitness(self):
        for individual in self.population:
            if individual.strategy == "cooperate":
                individual.fitness = individual.endowment - self.contribution_cost
            else:
                individual.fitness = individual.endowment

class Individual:
    def init(self, wealth, endowment):
        self.wealth = wealth
        self.endowment = endowment
        self.strategy = "cooperate" if random.uniform(0, 1) < 0.5 else "defect"
        self.fitness = 0


import numpy as np

class MarkovProcess:

    def init(self, states, transition_matrix, time_step):
        self.states = states
        self.transition_matrix = transition_matrix
        self.time_step = time_step
        self.prob_distribution = np.zeros(len(self.states))
        self.prob_distribution[0] = 1

    def update(self):
        new_prob_distribution = np.zeros(len(self.states))
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                new_prob_distribution[i] += self.transition_matrix[j][i] * self.prob_distribution[j] * self.time_step
                new_prob_distribution[j] -= self.transition_matrix[j][i] * self.prob_distribution[j] * self.time_step

        self.prob_distribution = new_prob_distribution


# Try to replicate some of Vasconcelos et al 2014: Climate policies under wealth inequality
# NOTE to self: the reason the p values sometimes don't line up with the gradient field is because the corners and edges are sticky
# homophily makes the edges sticky

from scipy.special import comb as comb
import numpy as np
from scipy.linalg import eig as eig

# parameters

h = 0  # homophily parameter (0≤h≤1), such that when h=1, individuals are restricted to influence (and be influenced) by those of the same wealth status, whereas when h=0, no wealth discrimination takes place
r = 0.2  # the perception of risk of collective disaster, individuals in the group will lose whatever they have if the target is not met
beta = 3  # intensity of selection

Z = 200  # total population with 4 different subpopulations, rich cooperators (ir), rich defectors (ZR - ir), poor cooperators (ip), poor defectors (ZP - ip)
ZR = 40  # number of rich in the population
ZP = 160  # number of poor in the population
N = 6  # game group size

c = 0.1  # contribution of the endowment to the target
bP = 0.625  # endowments of the poor
bR = 2.5  # endowments of the rich

cR = c * bR  # contribution of the rich cooperator to the target
cP = c * bP  # contribution of the poor cooperator to the target

cbar = (ZR * bR + ZP * bP) / Z  # average endowment of the population

M = 3  # integer used to calculate the target (target = M*c*cbar)

# calculate secondary parameters
# ---


f = 3  # factor to apply to mu
mu = f * 1 / Z  # mutation probability μ, individuals adopt a randomly chosen different strategy, in such a way that when μ= 1, the individual does change strategy

# define handy functions
# ---

# Theta_fnc = lambda Delta: 1 if Delta >= 0 else 0
# Delta_fnc = lambda jR, jP: cR*jR + cP*jP - M*c*cbar
Theta_fnc = lambda jR, jP: 1 if cR * jR + cP * jP - M * c * cbar >= 0 else 0

# payoffs functions :
# the payoff of an individual playing in a group in which there are jR rich cooperators, jP poor cooperators,
PiDR = lambda jR, jP: bR * (Theta_fnc(jR, jP) + (1 - r) * (1 - Theta_fnc(jR, jP)))
PiDP = lambda jR, jP: bP * (Theta_fnc(jR, jP) + (1 - r) * (1 - Theta_fnc(jR, jP)))
PiCR = lambda jR, jP: bR * (Theta_fnc(jR, jP) + (1 - r) * (1 - Theta_fnc(jR, jP))) - cR
PiCP = lambda jR, jP: bP * (Theta_fnc(jR, jP) + (1 - r) * (1 - Theta_fnc(jR, jP))) - cP

# fitness functions :
# fitness fXk of an individual adopting a given strategy X (cooperate(C) or defect(D) in a population of wealth class k will be associated with the average payoff of that strategy in the
# entire population. This can be computed for a given configuration of strategies and wealth classes specified by i={ir, ip}, using
# a multivariate hypergeometric sampling

fCR = lambda iR, iP: (1 / comb(Z - 1, N - 1)) * \
                     sum(sum(
                         comb(iR - 1, jR) * comb(iP, jP) * comb(Z - iR - iP, N - 1 - jR - jP) * PiCR(jR + 1, jP)
                         for jP in range(N - jR)) for jR in range(N))

fCP = lambda iR, iP: (1 / comb(Z - 1, N - 1)) * \
                     sum(sum(
                         comb(iR, jR) * comb(iP - 1, jP) * comb(Z - iR - iP, N - 1 - jR - jP) * PiCP(jR, jP + 1)
                         for jP in range(N - jR)) for jR in range(N))

fDR = lambda iR, iP: (1 / comb(Z - 1, N - 1)) * \
                     sum(sum(
                         comb(iR, jR) * comb(iP, jP) * comb(Z - 1 - iR - iP, N - 1 - jR - jP) * PiDR(jR, jP)
                         for jP in range(N - jR)) for jR in range(N))

fDP = lambda iR, iP: (1 / comb(Z - 1, N - 1)) * \
                     sum(sum(
                         comb(iR, jR) * comb(iP, jP) * comb(Z - 1 - iR - iP, N - 1 - jR - jP) * PiDP(jR, jP)
                         for jP in range(N - jR)) for jR in range(N))


def fXk(iR, iP, X, k):

    if X == 'C' and k == 'R':
        res = fCR(iR, iP)
    elif X == 'C' and k == 'P':
        res = fCP(iR, iP)
    elif X == 'D' and k == 'R':
        res = fDR(iR, iP)
    elif X == 'D' and k == 'P':
        res = fDP(iR, iP)
    else:
        res = None

    return (res)


# Fermi function
Fe = lambda iR, iP, X1, k1, X2, k2: 1 + np.exp(beta * (fXk(iR, iP, X1, k1) - fXk(iR, iP, X2, k2)))


def T(iR, iP, X, k):
    '''
    Probability of transition of a k-wealth individual (k in {R, P}) from strategy
    X (X in {C, D}) to Y (opposite of X)
    '''

    Y = 'C' if X == 'D' else 'D'
    l = 'R' if k == 'P' else 'P'

    Zk = ZR if k == 'R' else ZP
    Zl = ZR if l == 'R' else ZP

    ik = iR if k == 'R' else iP
    iXk = ik if X == 'C' else Zk - ik

    il = iR if l == 'R' else iP
    iYl = il if Y == 'C' else Zl - il
    iYk = ik if Y == 'C' else Zk - ik

    TXYk = (iXk / Z) * (mu + (1 - mu) * (
            iYk / ((Zk - 1 + (1 - h) * Zl) * Fe(iR, iP, X, k, Y, k)) + \
            (1 - h) * iYl / ((Zk - 1 + (1 - h) * Zl) * Fe(iR, iP, X, k, Y, l))
    ))

    return (TXYk)


# enumerate all possible states (iR, iP)
# ---

iV = [(iR, iP) for iP in range(ZP + 1) for iR in range(ZR + 1)]  # list of states
i2idx = {i: idx for idx, i in enumerate(iV)}  # reverse dictionary from state to index
len_iV = len(iV)

# create W, grad
# ---

W = np.zeros((len_iV, len_iV))
grad_iR = np.zeros((ZP + 1, ZR + 1))  # rich on the x-axis
grad_iP = np.zeros((ZP + 1, ZR + 1))
stay_iR = np.zeros((ZP + 1, ZR + 1))  # rich on the x-axis
stay_iP = np.zeros((ZP + 1, ZR + 1))
go_iR = np.zeros((ZP + 1, ZR + 1))  # rich on the x-axis
go_iP = np.zeros((ZP + 1, ZR + 1))

# each state can only ever transition in one of four ways:
# iR -> iR+1, iR -> iR-1, iP -> iP+1, iP -> iP-1,
# or it stays the same

for idx, i in enumerate(iV):

    iR, iP = i

    # calculate probabilities of each and population the matrix

    if iR < ZR:
        TiR_gain = T(iR, iP, 'D', 'R')
        W[i2idx[(iR + 1, iP)], idx] = TiR_gain
    else:
        TiR_gain = 0

    if iR > 0:
        TiR_loss = T(iR, iP, 'C', 'R')
        W[i2idx[(iR - 1, iP)], idx] = TiR_loss
    else:
        TiR_loss = 0

    if iP < ZP:
        TiP_gain = T(iR, iP, 'D', 'P')
        W[i2idx[(iR, iP + 1)], idx] = TiP_gain
    else:
        TiP_gain = 0

    if iP > 0:
        TiP_loss = T(iR, iP, 'C', 'P')
        W[i2idx[(iR, iP - 1)], idx] = TiP_loss
    else:
        TiP_loss = 0

    W[(idx, idx)] = 1 - TiR_gain - TiR_loss - TiP_gain - TiP_loss

    grad_iR[iP, iR] = TiR_gain - TiR_loss
    grad_iP[iP, iR] = TiP_gain - TiP_loss

    stay_iR[iP, iR] = 1 - TiR_gain - TiR_loss
    stay_iP[iP, iR] = 1 - TiP_gain - TiP_loss

    go_iR[iP, iR] = TiR_gain + TiR_loss
    go_iP[iP, iR] = TiP_gain + TiP_loss

# get relative proportions of time spent in different states
# ---

eigs, leftv, rightv = eig(W, left=True, right=True)
domIdx = np.argmax(np.real(eigs))  # index of the dominant eigenvalue
L = np.real(eigs[domIdx])  # the dominant eigenvalue
p = np.real(rightv[:, domIdx])  # the right-eigenvector is the relative proportions in classes at ss
p = p / np.sum(p)  # normalise it
