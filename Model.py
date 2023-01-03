""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""

import numpy as np
import PGGStrategy
import Player


class ClimateGame:

    def __init__(self, popuplation_size: int, group_size: int, nb_rich: int, strategies: list, profiles: list,
                 fraction_endowment: float, homophily:float, risk:float, M:float) -> None:
        self.population_size = popuplation_size  # Z
        self.group_size = group_size  # N
        self.rich = nb_rich  # Zr
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

        self.nb_group_configurations_ = 3  # TMP
        self.payoffs_ = np.zeros(4)

        #  Initialize payoff matrix
        self.calculate_payoffs()



    def calculate_payoffs(self) -> np.ndarray:
        """
        :return: payoff array Dp, Dr, Cp, Cr
        """

        payoff_container = np.zeros(4)  # 4 different strategies Dp, Dr, Cp, Cr
        for i in range(self.nb_group_configurations_):
            # Get group composition
            group_composition = sample_simplex(i, self.group_size, self.nb_strategies)
            self.play(group_composition, payoff_container)
            for strategy_index, strategy_payoff in enumerate(payoff_container):
                #self.update_payoff(strategy_index, i, strategy_payoff) # TODO ?
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs()

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

    def calculate_fitness(self, strategy_index: int, pop_size: int, population_state: np.ndarray) -> float:
        """
        This method should return the fitness of strategy
        with index `strategy_index` for the given `population_state`.
        """

    pass

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
    player_p = Player.Player(0, 0.625)
    player_r = Player.Player(1, 2.5)

    strategy_defect = PGGStrategy.PGGStrategy(0)
    strategy_coop = PGGStrategy.PGGStrategy(1)

    population_size = 200
    group_size = 6
    nb_rich = 40
    strategies = [strategy_defect, strategy_coop]
    profiles = [player_p, player_r]
    fraction_endowment = 0.1
    homophily = 0.5
    risk = 0.1
    M = 3  # Between 0 and group_size

    Game = ClimateGame(popuplation_size= population_size, group_size= group_size,  nb_rich= nb_rich, strategies= strategies,
                       profiles= profiles, fraction_endowment= fraction_endowment, homophily= homophily, risk= risk, M= M)

    print(Game.payoffs_)
