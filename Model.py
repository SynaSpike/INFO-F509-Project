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

    def __init__(self, group_size: int, nb_rich: int, strategies: list, profiles: list,
                 fraction_endowment: float, poor_coop: float, rich_coop: float) -> None:

        self.group_size = group_size  # Z
        self.rich = nb_rich
        self.poor = group_size - nb_rich
        self.strategies = strategies
        self.profile = profiles
        self.fraction_endowment = fraction_endowment
        self.poor_coop = poor_coop
        self.rich_coop = rich_coop


    def payoff(self, defect: bool) -> np.ndarray:

        payoff = np.zeros(4)
        if defect:

            payoff[0] = self.profile[0].endowment*4
            payoff[1] = self.profile[1].endowment*2
        else:
            payoff[2] = self.profile[0].endowment*1
            payoff[3] = self.profile[1].endowment*0.5

        return payoff


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
        pass

    def calculate_payoffs(self) -> np.ndarray:
        """
        This method should set a numpy.ndarray called self.payoffs_ with the
        expected payoff of each strategy in each possible
        state of the game
        """

    pass


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


    #def payoff(self, strategy: int, group_configuration: list[int]) -> float:
        #"""
        #This method should return the payoff of a strategy
        #for a given `group_configuration`, which gives
        #the counts of each strategy in the group.
        #This method only needs to be implemented for N-player games
        #"""


    #pass


    def save_payoffs(self, file_name: str) -> None:
        """
        This method should implement a mechanism to save
        the payoff matrix and parameters of the game to permanent storage.
        """


    pass


if __name__ == '__main__':

    player_p = Player.Player(0, 10)
    player_r = Player.Player(1, 20)

    strategy_defect = PGGStrategy.PGGStrategy(0)
    strategy_coop = PGGStrategy.PGGStrategy(1)

    group_size = 6
    nb_rich = 3
    strategies = [strategy_defect, strategy_coop]
    profiles = [player_p, player_r]
    fraction_endowment = 1
    poor_coop = 0.5
    rich_coop = 0.5

    Game = ClimateGame (group_size, nb_rich, strategies, profiles,
                 fraction_endowment, poor_coop, rich_coop)

    print(Game.payoff(True))