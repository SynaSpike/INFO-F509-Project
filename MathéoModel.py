""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""

from egttools import (calculate_nb_states, calculate_state, sample_simplex, )
from egttools.games import AbstractNPlayerGame
from egttools.behaviors import pgg_behaviors
from numpy import ndarray
from scipy.special import binom
import math
from . import PGGStrategy
import random



class PGG(AbstractNPlayerGame):

    def __init__(self, population:int, population_rich:int, population_poor:int,
                 group_size:int, endowment_rich:float, endowment_poor:float,
                 participation:float, risk:float, homophily:float, m:int, strategies:list[PGGStrategy]):

        #mu, beta, gradients selection  # TODO ?

        self.Z = population
        self.Zr = population_rich
        self.Zp = population_poor
        self.N = group_size
        self.bp = endowment_poor
        self.br = endowment_rich
        self.c = participation
        self.cr = self.br * self.c
        self.cp = self.bp * self.c
        self.b_bar = (self.Zr * self.br + self.Zp * self.bp)/self.Z
        self.r = risk
        self.h = homophily
        self.M = m
        self.Mcb_bar = self.M * self.c * self.b_bar

        self.nb_group_configurations = self.nb_group_configurations() # TODO ?
        self.strategies = strategies
        self.nb_strategies = len(strategies)

        super().__init__(len(self.strategies), group_size)
        self.calculate_payoffs()

    def payoff(self, strategy_index:int, group_composition:list[int]):
        '''
        Returns the payoff of a strategy given a group composition.
        If the group composition does not include the strategy, the payoff should be zero.

        :param strategy_index: The index of the strategy used by the player.
        :param group_composition: List with the group composition. The structure of this
        list depends on the particular implementation of this abstract method.

        :return: The payoff value. (float)
        '''

        # numer of rich and poor cooperators ?
        # TODO

        jr = 0
        jp = 0

        # rich defectors
        k = self.cr * jr + self.cp * jp - self.Mcb_bar
        Θ = k > 0 and 1 or 0
        total = Θ + (1-self.r) * (1-Θ)

        payoff_rich_D = self.br * total
        payoff_poor_D = self.bp * total
        payoff_rich_C = payoff_rich_D - self.cr
        payoff_poor_C = payoff_rich_D - self.cp

        # return the corresponding payoff to the strategy
        # TODO

    def calculate_payoffs(self):
        '''
        Estimates the payoffs for each strategy and returns the values in a matrix.

        Each row of the matrix represents a strategy and each column a game state. E.g.,
        in case of a 2 player game, each entry a_ij gives the payoff for strategy i against strategy j.
        In case of a group game, each entry a_ij gives the payoff of strategy i for game state j,
        which represents the group composition.

        :return: A matrix with the expected payoffs for each strategy given each possible game state.
        (numpy.ndarray[numpy.float64[m, n]])
        '''
        self.payoffs = ndarray(shape=(2,2), dtype=float)

    def calculate_fitness(self, strategy_index, population, strategies):

        # ir and ip ?
            # TODO
        ir = 0
        ip = 0

        # rich cooperators
        sum_1 = 0

        for jr in range(self.N):
            sum_2 = 0
            for jp in range(self.N-jr):
                sum_2 += binom(ir-1, jr) * binom(ip, jp) * binom(self.Z-ir-ip, self.N-1-jr-jp) * self.payoff(strategy_index)
            sum_1 += sum_2

        fit_r_c = binom(self.Z-1, self.N-1)^(-1) * sum_1

        # rich defectors
            # TODO
        # poor cooperators
            # TODO
        # poor defectors
            # TODO

        # return the fitness
            # TODO

    def create_pop(self):
        num_rich = int(self.population_size * self.wealth_ratio)
        num_poor = self.population_size - num_rich
        self.population = []

        for i in range(num_rich):
            self.population.append(PGGStrategy("rich"))
        for i in range(num_poor):
            self.population.append(["poor", random.randint(1, 2) == 1 and "C" or "D"])

        random.shuffle(self.population)
        groups = [self.population[i:i + self.group_size] for i in range(0, len(self.population), self.group_size)]

    def transition(self):

        import math

        mu = 0
        beta = 0
        ikX = 0
        ikY = 0
        ilY = 0
        Z = 0
        h = 0
        Zk = 0
        Zl = 0
        fkX = 0
        fkY = 0
        flY = 0

        fermi_1 = (1 + math.e ** (beta * (fkX - fkY))) ** -1
        fermi_2 = (1 + math.e ** (beta * (fkX - flY))) ** -1
        param1 = ikY / (Zk - 1 + (1 - h) * Zl)
        param2 = ((1 - h) * ilY) / (Zk - 1 + (1 - h) * Zl)
        Tkx_to_y = (ikX / Z) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

    def play(self, group_composition:list(int), game_payoffs:list(float)):
        '''
        Updates the vector of payoffs with the payoffs of each player after playing the game.

        This method will run the game using the players and player types defined in :param group_composition,
        and will update the vector :param game_payoffs with the resulting payoff of each player.

        :param group_composition: A list with counts of the number of players of each strategy in the group.
        :param game_payoffs: A list used as container for the payoffs of each player
        '''

        contributions = 0.0
        non_zero = []
        for i, strategy_count in enumerate(group_composition):
            if strategy_count == 0:
                continue
            else:
                non_zero.append(i)
                action = self.strategies[i].get_action()
                if action == 1: # 1 = coop
                    # add to the contribution if they are rich or poor
                    contributions += strategy_count * self.c_
                    game_payoffs[i] = - self.c_

        # check if threshold is obtained if not then payoff is 0 for everyone

        benefit = (contributions * self.r_) / self.group_size_
        game_payoffs[non_zero] += benefit


# https://egttools.readthedocs.io/en/master/tutorials/analytical_methods.html

## calculate_fixation_probability(invading_strategy_index, resident_strategy_index:, beta:):
# Calculates the fixation probability of a single mutant of an invading strategy of index invading_strategy_index in a
# population where all individuals adopt the strategy with index resident_strategy_index. The parameter beta gives the intensity of selection.
#
## calculate_transition_and_fixation_matrix_sml(beta):
# Calculates the transition and fixation matrices assuming the SML. Beta gives the intensity of selection.
#
## calculate_gradient_of_selection(beta, state):
# Calculates the gradient of selection (without considering mutation) at a given population state. The state parameter
# must be an array of shape (nb_strategies,), which gives the count of individuals in the population adopting each strategy.
# This method returns an array indicating the gradient in each direction. In this stochastic model, gradient of selection
# means the most likely path of evolution of the population.
#
## calculate_transition_matrix(beta, mu):
# Calculates the transition matrix of the Markov chain that defines the dynamics of the population. beta gives the
# intensity of selection and mu the mutation rate.

