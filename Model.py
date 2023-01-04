""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""
import random
from scipy.special import binom

import egttools
import numpy as np
import PGGStrategy
import Player


class ClimateGame:

    def __init__(self, popuplation_size: int, group_size: int, nb_rich: int, strategies: list, profiles: list,
                 fraction_endowment: float, homophily:float, risk:float, M:float, rich_end:int, poor_end:int) -> None:
        self.population_size = popuplation_size  # Z
        self.group_size = group_size  # N
        self.nb_group = self.population_size // self.group_size
        self.rich = nb_rich  # Zr
        self.rich_end = rich_end
        self.poor_end = poor_end
        self.poor = popuplation_size - nb_rich  # Zp
        self.strategies = strategies  # Ds or Cs
        self.profiles = profiles  # Poor or Rich
        self.nb_strategies = 4  # Dp Dr Cp Cr
        self.fraction_endowment = fraction_endowment  # C
        self.poor_coop = profiles[0].endowment * fraction_endowment
        self.rich_coop = profiles[1].endowment * fraction_endowment
        self.homophily = homophily  # h
        self.risk = risk  # r
        self.treshold = M * fraction_endowment * ((self.poor_coop * self.poor) * (self.rich_coop * self.rich) /
                                                  self.population_size)
        self.rich_pg = (self.rich / self.population_size) *  self.group_size  # Rich per group
        self.poor_pg = self.group_size - self.rich_pg   # Poor per group

        self.nb_group_configurations_ = egttools.calculate_nb_states(self.group_size, self.nb_strategies)
        self.payoffs_ = np.zeros(shape=(self.nb_strategies,self.nb_group))

        self.groups = self.sample(nb_rich / popuplation_size)

        #  Initialize payoff matrix
        self.calculate_payoffs()

    def sample(self, wealth_ratio):
        """
        :return: np.ndarray shape(nb.groups,2) of number of poor and rich.
        """

        num_rich = int(self.population_size * wealth_ratio)
        num_poor = self.population_size - num_rich
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

        game_payoffs[2] = game_payoffs[0] - self.poor_coop * self.poor_pg
        game_payoffs[3] = game_payoffs[1] - self.rich_coop * self.rich_pg

        return game_payoffs

    def calculate_fitness(self) -> float:
        """
        # strategy_index: int, pop_size: int, population_state: np.ndarray
        This method should return the fitness of strategy
        with index `strategy_index` for the given `population_state`.
        """

        ir = self.rich # ?? not Zr
        ip = self.population_size - self.rich

        # rich cooperators
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size - jr):
                rich_payoff = self.play([0, 0, jp, jr + 1], np.zeros(4))[3]
                # Do not care about the nbr of defector (does not affect payoff
                sum_2 += binom(ir - 1, jr) * binom(ip, jp) * binom(self.population_size - ir - ip,
                                                                   self.group_size - 1 - jr - jp) * rich_payoff
            sum_1 += sum_2

        fit_r_c = binom(self.population_size - 1, self.group_size - 1)**(-1) * sum_1

        return fit_r_c

        # rich defectors
        # TODO
        # poor cooperators
        # TODO
        # poor defectors
        # TODO

        # return the fitness
        # TODO

    def __str__(self) -> str:
        """
        This method should return a string representation of the game.
        """
        return "ClimateGame Object"

    pass

    def nb_strategies(self) -> int:
        """
        This method should return the number of strategies which can play the game.
        """

    pass

    def type(self) -> str:
        """
        This method should return a string representing the type of game.
        """

    pass

    def payoffs(self) -> np.ndarray:
        """
        This method should return the payoff matrix of the game,
        which gives the payoff of each strategy
        in each given context.
        """

    pass

    # def payoff(self, strategy: int, group_configuration: list[int]) -> float:
    # """
    # This method should return the payoff of a strategy
    # for a given `group_configuration`, which gives
    # the counts of each strategy in the group.
    # This method only needs to be implemented for N-player games
    # """

    # pass

    def save_payoffs(self, file_name: str) -> None:
        """
        This method should implement a mechanism to save
        the payoff matrix and parameters of the game to permanent storage.
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
    nb_rich = 40

    profiles = [player_p, player_r]
    fraction_endowment = 0.1
    homophily = 0.5
    risk = 0.1
    M = 3  # Between 0 and group_size

    Game = ClimateGame(popuplation_size= population_size, group_size= group_size,  nb_rich= nb_rich, strategies= strategies,
                       profiles= profiles, fraction_endowment= fraction_endowment, homophily= homophily, risk= risk, M= M,
                       rich_end= 2.5, poor_end= 0.625)

    #print(len(Game.sample(0.8)[32]))

    #for i in range(6):
        #print(Game.sample(0.8)[32][i].get_wealth())

    #a = Game.sample(0.8)[32]
    #print(len(a))
    #print(Game.get_comp(a))

    print(Game.payoffs_[:,32])

    print(Game.calculate_fitness())


