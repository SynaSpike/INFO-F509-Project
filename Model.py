""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""
import random
import time
import tqdm

from scipy.special import binom

import egttools
import numpy as np
import PGGStrategy
import Player


class ClimateGame:


    def __init__(self, popuplation_size: int, group_size: int, nb_rich: int, nb_poor:int, strategies: list, profiles: list,
                 fraction_endowment: float, homophily:float, risk:float, M:float, rich_end:int, poor_end:int) -> None:
        self.population_size = popuplation_size  # Z
        self.group_size = group_size  # N
        self.nb_group = self.population_size // self.group_size
        self.rich = nb_rich  # Zr
        self.rich_end = rich_end
        self.poor_end = poor_end
        self.poor = nb_poor  # Zp
        self.strategies = strategies  # Ds or Cs
        self.profiles = profiles  # Poor or Rich
        self.nb_strategies = 4  # Dp Dr Cp Cr
        self.fraction_endowment = fraction_endowment  # C
        self.poor_coop = profiles[0].endowment * fraction_endowment # Cooperation of the Poor
        self.rich_coop = profiles[1].endowment * fraction_endowment # Cooperation of the rich
        self.homophily = homophily  # h
        self.risk = risk  # r
        self.treshold = M * fraction_endowment * ((((profiles[0].endowment * self.poor) + (profiles[1].endowment * self.rich)) /self.population_size))
        self.rich_pg = (self.rich / self.population_size) *  self.group_size  # Rich per group
        self.poor_pg = self.group_size - self.rich_pg   # Poor per group

        self.nb_group_configurations_ = egttools.calculate_nb_states(self.group_size, self.nb_strategies)
        self.payoffs_ = np.zeros(shape=(self.nb_strategies,self.nb_group))

        self.population = [] #  Player population
        self.groups = self.sample(nb_rich / popuplation_size)

        #  Initialize payoff matrix
        self.calculate_payoffs()

    def sample(self, wealth_ratio):
        """
        :return: np.ndarray shape(nb.groups,2) of number of poor and rich.
        """

        num_rich = self.rich
        num_poor = self.poor
        self.population = []

        for i in range(num_rich):
            self.population.append(Player.Player(wealth = 1, endowment= self.rich_end, strategy= random.choice(self.strategies)))
        for i in range(num_poor):
            self.population.append(Player.Player(wealth = 0, endowment= self.poor_end, strategy= random.choice(self.strategies)))

        random.shuffle(self.population)

        groups = [self.population[i:i + self.group_size] for i in range(0, self.group_size*self.nb_group, self.group_size)]

        return groups


    def get_comp(self, group):

        comp = np.zeros(self.nb_strategies)

        for plr in group:

            if plr.wealth == 0 and plr.strategy.action == 0:
                comp[0] += 1
            if plr.wealth == 1 and plr.strategy.action == 0:
                comp[1] += 1
            if plr.wealth == 0 and plr.strategy.action == 1:
                comp[2] += 1
            if plr.wealth == 1 and plr.strategy.action == 1:
                comp[3] += 1

        return comp

    def get_nbr_cat(self, population: list):
        """
        :param population: List of Player population
        :return: the number of (wealth; strat)
        """

        Dp = 0
        Dr = 0
        Cp = 0
        Cr = 0

        for plr in range(population):

            if plr.wealth == 0 and plr.strategy.action == 0:
                Dp += 1
            if plr.wealth == 1 and plr.strategy.action == 0:
                Dr += 1
            if plr.wealth == 0 and plr.strategy.action == 1:
                Cp += 1
            if plr.wealth == 1 and plr.strategy.action == 1:
                Cr += 1

        return [Dp, Dr, Cp, Cr]


    def calculate_payoffs(self) -> np.ndarray:
        """
        :return: payoff array Dp, Dr, Cp, Cr
        """
        payoff_container = np.zeros(self.nb_strategies)  # 4 different strategies Dp, Dr, Cp, Cr

        for i in range(self.nb_group):

            group_composition = self.get_comp(self.groups[i])

            # Get group composition
            self.play(group_composition, payoff_container) # Update the container with the new payoff following group_comp


            for strategy_index, strategy_payoff in enumerate(payoff_container):
                self.payoffs_[strategy_index, i] = strategy_payoff


            # Reinitialize payoff vector
            payoff_container[:] = 0
        return self.payoffs_

    def play(self, group_composition, game_payoffs: np.ndarray) -> None:
        """
        Calculates the payoff of each strategy inside the group.

        Parameters
        ----------
        group_composition: Union[List[int], numpy.ndarray]
            counts of each strategy inside the group.
        game_payoffs: numpy.ndarray
            container for the payoffs of each strategy
        """

        k = self.rich_coop * group_composition[3] + self.poor_coop * group_composition[2] - self.treshold

        if k >= 0:
            heav = 1

        else:
            heav = 0


        game_payoffs[0] = self.profiles[0].endowment * (heav + (1 - self.risk) * (1 - heav))
        game_payoffs[1] = self.profiles[1].endowment * (heav + (1 - self.risk) * (1 - heav))

        game_payoffs[2] = game_payoffs[0] - self.poor_coop
        game_payoffs[3] = game_payoffs[1] - self.rich_coop

        return game_payoffs

    def calculate_fitness(self, ir, ip) -> float:
        """
        # strategy_index: int, pop_size: int, population_state: np.ndarray
        This method should return the fitness of strategy
        with index `strategy_index` for the given `population_state`.
        :param: ir nbr of rich
        :param: ip nbr of poor
        :format:
        [ Dp, Dr, Cp, Cr ]

        """
    # rich cooperators
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):
                payoff = self.play([0, 0, jp, jr + 1], np.zeros(4))[3] # Rich coop
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += binom(ir - 1, jr) * binom(ip, jp) * binom(self.population_size - ir - ip,
                                                                   self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RC = binom(self.population_size - 1, self.group_size - 1)**(-1) * sum_1


    # rich defectors
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):
                payoff = self.play([0, 0, jp, jr], np.zeros(4))[1]  # Rich defector
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += binom(ir, jr) * binom(ip, jp) * binom(self.population_size - 1 - ir - ip,
                                                                   self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RD = binom(self.population_size - 1, self.group_size - 1)**(-1) * sum_1

    # poor cooperators
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):
                payoff = self.play([0, 0, jp + 1, jr], np.zeros(4))[2]  # Poor coop
                # Do not care about the nbr of defector (does not affect payoff
                sum_2 += binom(ir, jr) * binom(ip - 1, jp) * binom(self.population_size - ir - ip,
                                                               self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PC = binom(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

    # poor defectors
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size-jr-1, -1, -1):

                payoff = self.play([0, 0, jp, jr], np.zeros(4))[0]  # Poor defector
                # Do not care about the nbr of defector (does not affect payoff
                sum_2 += binom(ir, jr) * binom(ip, jp) * binom(self.population_size - 1 - ir - ip,
                                                                   self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PD = binom(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        return [PD, RD, PC, RC]


    def transition_probabilities(self, ir, ip, rounding:bool):
        """
        This function is used to return T
        :return: T prob
        :format:
        --------------
       Dp |Dp Cp Dr Cr |
       Cp |Dp Cp Dr Cr |
       Dr |Dp Cp Dr Cr |
       Cr |Dp Cp Dr Cr |
        -------------
        """
        fit = self.calculate_fitness(ir, ip) # Dp Dr Cp Cr
        Dp = self.poor - ip # Nbr of poor defector
        Dr = self.rich - ir # Nbr of rich defector
        beta = 5.0
        mu = 1/self.population_size


        T_prob = np.zeros(shape=(self.nb_strategies, self.nb_strategies))

        # Transition Cp -> Dp
        fermi_1 = (1 + np.exp(beta * (fit[2]- fit[0])))** -1 # Cp -> Dp
        fermi_2 = (1 + np.exp((beta * (fit[2] - fit[1])))) ** -1 # Cp -> Dr
        param1 = (Dp / (self.poor - 1 + (1 - self.homophily)* self.rich))
        param2 =  ((1 - self.homophily) * Dr) / (self.poor - 1 + (1 - self.homophily) * self.rich)
        T_prob[1,0] = ( ip / self.population_size ) * ( ( 1 - mu) * ( param1 * fermi_1  + param2 * fermi_2 ) + mu )

        # Transition Cr -> Dr
        fermi_1 = (1 + np.exp(beta * (fit[3] - fit[1]))) ** -1  # Cr -> Dr
        fermi_2 = (1 + np.exp((beta * (fit[3] - fit[0])))) ** -1  # Cr -> Dp
        param1 = (Dr / (self.rich - 1 + (1 - self.homophily) * self.poor))
        param2 = ((1 - self.homophily) * Dp) / (self.rich - 1 + (1 - self.homophily) * self.poor)
        T_prob[3,2] = (ir / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dp -> Cp
        fermi_1 = (1 + np.exp(beta * (fit[0] - fit[2]))) ** -1  # Dp -> Cp
        fermi_2 = (1 + np.exp((beta * (fit[0] - fit[3])))) ** -1  # Dp -> Cr
        param1 = (ip / (self.poor - 1 + (1 - self.homophily) * self.rich))
        param2 = ((1 - self.homophily) * ir) / (self.poor - 1 + (1 - self.homophily) * self.rich)
        T_prob[0,1] = (Dp / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

        # Transition Dr -> Cr
        fermi_1 = (1 + np.exp(beta * (fit[1] - fit[3]))) ** -1  # Dr -> Cr
        fermi_2 = (1 + np.exp((beta * (fit[1] - fit[2])))) ** -1  # Dr -> Cp
        param1 = (ir / (self.rich - 1 + (1 - self.homophily) * self.poor))
        param2 = ((1 - self.homophily) * ip) / (self.rich - 1 + (1 - self.homophily) * self.poor)
        T_prob[2,3] = (Dr / self.population_size) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)


        T_prob[0:1,2:3] = 0 # Poor -> Rich
        T_prob[2:3, 0:1] = 0 # Rich -> Poor
        T_prob[0,0] = 1 - T_prob[0,1]
        T_prob[1,1] = 1 - T_prob[1,0]
        T_prob[2, 2] = 1 - T_prob[2, 3]
        T_prob[3, 3] = 1 - T_prob[3, 2]

        if rounding:
            # Rounding 4 numbers
            for i in range(len(T_prob)):
                for j in range(len(T_prob[i])):
                    T_prob[i,j] = round(T_prob[i,j], 4)

        return T_prob

    def population_configuration(self):

        pop_conf = []

        for i in range(self.rich + 1):

            for j in range(self.poor + 1):

                pop_conf.append((i,j))

        return pop_conf

    def transition_matrix(self):

        matrix = []

        conf = self.population_configuration()


        for i in range(len(conf))  and tqdm.trange(len(conf)):
            m_tmp = []

            for j in range(len(conf)):

                #m_tmp.append([self.transition_probabilities(conf[j][0], conf[j][1],True)])
                self.transition_probabilities(conf[j][0], conf[j][1],True)

            #matrix.append(m_tmp)

        return matrix

    @staticmethod

    def fermi(beta:float, fitness_diff:float):

        return (1 + np.exp(beta*fitness_diff))**(-1)

    def __str__(self) -> str:
        """
        This method should return a string representation of the game.
        """
        return "ClimateGame Object"

    pass

    def type(self) -> str:
        """
        This method should return a string representing the type of game.
        """
        pass


if __name__ == '__main__':

    strategy_defect = PGGStrategy.PGGStrategy(0)
    strategy_coop = PGGStrategy.PGGStrategy(1)
    strategies = [strategy_defect, strategy_coop]

    player_p = Player.Player(0, 0.625, strategies)
    player_r = Player.Player(1, 2.5, strategies)

    population_size = 200
    group_size = 6
    nb_rich = 39
    nb_poor = 159

    profiles = [player_p, player_r]
    fraction_endowment = 0.25
    homophily = 0
    risk = 0.1
    M = 3  # Between 0 and group_size

    Game = ClimateGame(popuplation_size= population_size, group_size= group_size,  nb_rich= nb_rich, nb_poor=nb_poor, strategies= strategies,
                       profiles= profiles, fraction_endowment= fraction_endowment, homophily= homophily, risk= risk, M= M,
                       rich_end= 2.5, poor_end= 0.625)

    print("------------------------ PAYOFF ---------------------")
    print("PAYOFF REACHED TRESHOLD: ", Game.play([0,0,4,2], [0,0,0,0]), "\n")
    print("PAYOFF UNREACHED TRESHOLD: ", Game.play([0, 0, 0, 0], [0, 0, 0, 0]), "\n")
    print("------------------------ FITNESSES ---------------------")
    print(Game.calculate_fitness(39, 159), "\n")
    print("----------------------- TRANSITION PROB ----------------")
    print(Game.transition_probabilities(20, 60, rounding= False), "\n")
    print("----------------------- NUMBER OF CONFIG ----------------")
    print(len(Game.population_configuration()),"\n")
    print("----------------------- TRANSITION MATRIX ----------------")
    print(Game.transition_matrix())


