""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""
import random
from typing import Union, Any

from scipy.special import comb as comb

import egttools
import numpy as np
import PGGStrategy
import Player
import math
from scipy.linalg import eig as eig
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


class ClimateGame:

    def __init__(self, popuplation_size: int, group_size: int, nb_rich: int, fraction_endowment: float,
                 homophily: float,
                 risk: float, M: float, rich_endowment: int, poor_endowment: int, mu: float, beta: float,
                 rich_evolution: float, poor_evolution: float) -> None:

        self.population_size = popuplation_size  # Z
        self.group_size = group_size  # N
        self.nb_group = self.population_size // self.group_size
        self.rich = nb_rich  # Zr
        self.rich_endowment = rich_endowment
        self.poor_endowment = poor_endowment
        self.poor = popuplation_size - nb_rich  # Zp

        self.nb_strategies = 4  # Dp Dr Cp Cr
        self.fraction_endowment = fraction_endowment  # C
        self.poor_contribution = poor_endowment * fraction_endowment  # Contribution of the Poor
        self.rich_contribution = rich_endowment * fraction_endowment  # Contribution of the rich
        self.homophily = homophily  # h
        self.risk = risk  # r
        self.b_bar = ((poor_endowment * self.poor) + (rich_endowment * self.rich)) / self.population_size
        self.threshold = M * fraction_endowment * self.b_bar
        self.mu = mu
        self.beta = beta

        self.rich_evolution = rich_evolution
        self.poor_evolution = poor_evolution

    def play(self):
        populations_configurations = []
        populations_configurations_index = {}
        index = 0

        print("Setting up all possible population configuration...")
        for ip in range(self.poor + 1):
            for ir in range(self.rich + 1):
                populations_configurations.append((ir, ip))
                populations_configurations_index[(ir, ip)] = index
                index += 1

        self.W = np.zeros((index, index))
        self.populations_configurations = populations_configurations
        self.populations_configurations_index = populations_configurations_index
        self.populations_transitions_results = {}  # moyen de calculer les gradients de selections a partir de ca si nécessaire
        self.totalindex = index

        print("Calculating population transitions...")
        for index, pop_config in enumerate(self.populations_configurations):
            self.calculate_population_transitions(pop_config)

        print("Calculating eigen values...")
        eigs, leftv, rightv = eig(self.W, left=True, right=True)
        print("Getting the index of the dominant eigenvalue...")
        domIdx = np.argmax(np.real(eigs))  # index of the dominant eigenvalue
        print("Getting the dominant eigenvalue...")
        L = np.real(eigs[domIdx])  # the dominant eigenvalue
        print("Getting the right-eigenvector...")
        p = np.real(rightv[:, domIdx])  # the right-eigenvector is the relative proportions in classes at ss
        print("pmax =", max(p))
        print("Normalising the relative proportions...")

        p = p / np.sum(p)
        self.p = p  # normalise it
        print("pmax_norm =", max(p))

        print("Calculating ng...")
        ng = 0
        for index, P_bar_i in enumerate(p):
            ir, ip = self.populations_configurations[index]
            ng += P_bar_i * self.calculate_ag(ir, ip)

        print("ng:", ng)

        # 1 == (kX = rich defect -> kY = rich coop)
        # 2 == (kX = rich coop -> kY = rich defect)
        # 3 == (kX = poor defect -> kY = poor coop)
        # 4 == (kX = poor coop -> kY = poor defect)
        # 5 == (no transi)

        self.gradient_selection = [0 for i in range(self.totalindex)]
        self.gradient_rich = [0 for i in range(self.totalindex)]
        self.gradient_poor = [0 for i in range(self.totalindex)]

        for pop_config, result in dict.items(self.populations_transitions_results):
            ir, ip = pop_config
            index_ = self.populations_configurations_index[pop_config]
            self.gradient_selection[index_] = (result[0] - result[1], result[2] - result[3])
            self.gradient_rich[index_] = result[0] - result[1]
            self.gradient_poor[index_] = result[2] - result[3]

    def GraphStationaryDistribution(self):

        ZP = self.poor
        ZR = self.rich
        iV = self.populations_configurations
        grad_iR = self.gradient_rich
        grad_iP = self.gradient_poor

        P = np.zeros((ZP + 1, ZR + 1))  # rich on the x-axis
        for idx, pi in enumerate(self.p):
            iR, iP = iV[idx]
            P[iP, iR] = pi

        # plot
        # ---

        fig, ax = plt.subplots(figsize=(3, 6))

        iRV = list(range(ZR + 1))
        iPV = list(range(ZP + 1))

        im = ax.imshow(P, origin='lower', cmap='coolwarm_r', alpha=0.5)

        arrow_every = 5

        for index, gradient in enumerate(self.gradient_selection):
            if int(index//arrow_every) != index/arrow_every:
                continue
            ir, ip = self.populations_configurations[index]
            arrow = FancyArrowPatch((ir, ip), (ir + gradient[0] * 100, ip + gradient[1] * 100),
                                    arrowstyle='simple', color='k', mutation_scale=10)
            ax.add_patch(arrow)

        ax.set_xlim((-1, ZR + 1))
        ax.set_ylim((-1, ZP + 1))
        ax.set_xlabel(r'rich cooperators, $i_R$')
        ax.set_ylabel(r'poor cooperators, $i_P$')
        plt.axis('scaled')
        plt.tight_layout()
        plt.show()

    def GraphOnePopEvolution(self, evolving_population:str, ratio_cooperators:list, rich_endowments:list):

        evolv_pop_size = evolving_population == "R" and self.rich or self.poor

        x = []
        other_pop_size = self.population_size - evolv_pop_size

        for evolv_pop_coop in range(evolv_pop_size):
            x.append(evolv_pop_coop/evolv_pop_size)

        for rich_endowment in rich_endowments:
            self.rich_endowment = rich_endowment
            self.poor_endowment = (self.population_size - (rich_endowment * self.rich))/self.poor
            print(self.rich_endowment, self.poor_endowment)
            self.poor_contribution = poor_endowment * fraction_endowment  # Contribution of the Poor
            self.rich_contribution = rich_endowment * fraction_endowment  # Contribution of the rich
            self.b_bar = ((poor_endowment * self.poor) + (rich_endowment * self.rich)) / self.population_size
            self.threshold = M * fraction_endowment * self.b_bar

            self.play()

            evolv_pop_grad = evolving_population == "R" and self.gradient_rich or self.gradient_poor

            for ratio in ratio_cooperators:
                other_pop_coop = int(other_pop_size * ratio)
                y = []
                for evolv_pop_coop in range(evolv_pop_size):
                    pop_config = evolving_population == "R" and (evolv_pop_coop, other_pop_coop) or (other_pop_coop, evolv_pop_coop)
                    index_ = self.populations_configurations_index[pop_config]
                    y.append(evolv_pop_grad[index_])

                labeltext = str(ratio * 100)+"%"
                plt.plot(x, y, rich_endowment == rich_endowments[0] and '-' or '--', label=labeltext)

        plt.legend(loc='best')
        plt.show()

    def return_payoff(self, group_composition) -> None:
        """
        Calculates the payoff of each strategy inside the group.

        Parameters
        ----------
        group_composition: Union[List[int], numpy.ndarray]
            counts of each strategy inside the group.
        game_payoffs: numpy.ndarray
            container for the payoffs of each strategy
        """

        game_payoffs = np.zeros(self.nb_strategies)

        k = self.rich_contribution * group_composition[3] + self.poor_contribution * group_composition[
            2] - self.threshold

        heav = k >= 0 and 1 or 0

        game_payoffs[0] = self.poor_endowment * (heav + (1 - self.risk) * (1 - heav))
        game_payoffs[1] = self.rich_endowment * (heav + (1 - self.risk) * (1 - heav))

        game_payoffs[2] = game_payoffs[0] - self.poor_contribution
        game_payoffs[3] = game_payoffs[1] - self.rich_contribution

        return game_payoffs

    def calculate_fitness(self, ir, ip) -> float:
        """
        # strategy_index: int, pop_size: int, population_state: np.ndarray
        This method should return the fitness of strategy
        with index `strategy_index` for the given `population_state`.
        :param: ir nbr of rich
        :param: ip nbr of poor

        """
        # rich cooperators
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size - jr):
                payoff = self.return_payoff([0, 0, jp, jr + 1])[3]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir - 1, jr) * comb(ip, jp) * comb(self.population_size - ir - ip,
                                                                self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RC = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        # rich defectors
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size - jr):
                payoff = self.return_payoff([0, 0, jp, jr])[1]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir, jr) * comb(ip, jp) * comb(self.population_size - 1 - ir - ip,
                                                            self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        RD = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        # poor cooperators
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size - jr):
                payoff = self.return_payoff([0, 0, jp + 1, jr])[2]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir, jr) * comb(ip - 1, jp) * comb(self.population_size - ir - ip,
                                                                self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PC = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        # poor defectors
        sum_1 = 0

        for jr in range(self.group_size):
            sum_2 = 0
            for jp in range(self.group_size - jr):
                payoff = self.return_payoff([0, 0, jp, jr])[0]
                # Do not care about the nbr of defector (does not affect payoff)
                sum_2 += comb(ir, jr) * comb(ip, jp) * comb(self.population_size - 1 - ir - ip,
                                                            self.group_size - 1 - jr - jp) * payoff
            sum_1 += sum_2

        PD = comb(self.population_size - 1, self.group_size - 1) ** (-1) * sum_1

        return [PD, RD, PC, RC]

    def transition_probability(self, Zk, Zl, ikX, ikY, ilY, fkX, fkY, flY):
        # an individual with strategy X∈{C,D} in the subpopulation k∈{R,P} changes to a different strategy
        # Y∈{C,D}, both from the same subpopulation k and from the other population l
        # l = P if k = R, and l = R if k = P
        mu = self.mu
        beta = self.beta
        Z = self.population_size
        h = self.homophily
        # print("ikX:", ikX, "ikY:", ikY, "ilY:", ilY, "Zk:", Zk, "Zl:", Zl, "fkX:", fkX, "fkY:", fkY, "flY:", flY)

        fermi_1 = (1 + math.e ** (beta * (fkX - fkY))) ** -1
        fermi_2 = (1 + math.e ** (beta * (fkX - flY))) ** -1
        param1 = ikY / (Zk - 1 + (1 - h) * Zl)
        param2 = ((1 - h) * ilY) / (Zk - 1 + (1 - h) * Zl)
        # print(fermi_1, fermi_2, param1, param2)

        return (ikX / Z) * ((1 - mu) * (param1 * fermi_1 + param2 * fermi_2) + mu)

    def calculate_population_transitions(self, pop_config):
        ir, ip = pop_config
        index = self.populations_configurations_index[(ir, ip)]
        fitness = self.calculate_fitness(ir, ip)

        population_transitions = [
            (1, -1, 0, 0),  # kX = rich defect -> kY = rich coop
            (-1, 1, 0, 0),  # kX = rich coop -> kY = rich defect
            (0, 0, 1, -1),  # kX = poor defect -> kY = poor coop
            (0, 0, -1, 1),  # kX = poor coop -> kY = poor defect
            (0, 0, 0, 0)  # no transi
        ]

        transitions_results = {}

        for transition in population_transitions:
            ir_prime = pop_config[0] + transition[0]
            ip_prime = pop_config[1] + transition[2]
            result = 0

            if 0 <= ir_prime <= self.rich and 0 <= ip_prime <= self.poor:
                transition_index = self.populations_configurations_index[(ir_prime, ip_prime)]

                if transition == (1, -1, 0, 0):
                    # print("")
                    # print("Transition kX = rich defect -> kY = rich coop")
                    # k = rich, l = poor, X = defect, Y = coop
                    result = self.rich_evolution * self.transition_probability(Zk=self.rich, Zl=self.poor,
                                                                               ikX=self.rich - ir, ikY=ir, ilY=ip,
                                                                               fkX=fitness[1], fkY=fitness[3],
                                                                               flY=fitness[2])

                elif transition == (-1, 1, 0, 0):
                    # print("Transition kX = rich coop -> kY = rich defect")
                    # k = rich, l = poor, X = coop, Y = defect
                    result = self.rich_evolution * self.transition_probability(Zk=self.rich, Zl=self.poor, ikX=ir,
                                                                               ikY=self.rich - ir, ilY=self.poor - ip,
                                                                               fkX=fitness[3], fkY=fitness[1],
                                                                               flY=fitness[0])

                elif transition == (0, 0, 1, -1):
                    # print("Transition kX = poor defect -> kY = poor coop")
                    # k = pauvre, l = riche, X = defect, Y = coop
                    result = self.poor_evolution * self.transition_probability(Zk=self.poor, Zl=self.rich,
                                                                               ikX=self.poor - ip, ikY=ip, ilY=ir,
                                                                               fkX=fitness[0], fkY=fitness[2],
                                                                               flY=fitness[3])

                elif transition == (0, 0, -1, 1):
                    # print("Transition kX = poor coop -> kY = poor defect")
                    # k = pauvre, l = riche, X = coop, Y = defect
                    result = self.poor_evolution * self.transition_probability(Zk=self.poor, Zl=self.rich, ikX=ip,
                                                                               ikY=self.poor - ip, ilY=self.rich - ir,
                                                                               fkX=fitness[2], fkY=fitness[0],
                                                                               flY=fitness[1])

                elif transition == (0, 0, 0, 0):
                    # print("pas de transition")
                    result = 1 - sum(transitions_results.values())

                self.W[transition_index, index] = result

            transitions_results[len(transitions_results)] = result

        self.populations_transitions_results[(ir, ip)] = transitions_results

    def contribution_reached(self, jR, jP):
        if self.rich_contribution * jR + self.poor_contribution * jP >= self.threshold:
            return 1
        else:
            return 0

    def calculate_ag(self, iR, iP):
        # Multivariate hypergeometric sampling (fitness equations) to compute the (average) fraction of groups that
        # reach a total of Mcb in contributions
        Z = self.population_size
        N = self.group_size
        return sum(comb(iR, jR) * comb(iP, jP) * comb(Z - iR - iP, N - jR - jP) * self.contribution_reached(jR, jP)
                   for jR in range(N + 1) for jP in range(N + 1)) / comb(Z, N)

    @staticmethod
    def __str__(self) -> str:
        """
        This method should return a string representation of the game.
        """
        return "ClimateGame Object"


if __name__ == '__main__':
    population_size = 200
    nb_rich = 100
    group_size = 6
    rich_endowment = 1.35
    poor_endowment = 0.65

    fraction_endowment = 0.1
    homophily = 0
    risk = 0.3
    M = 3  # Between 0 and group_size

    mu = 1 / population_size
    beta = 10

    rich_evolution = 0
    poor_evolution = 1

    Game = ClimateGame(popuplation_size=population_size, group_size=group_size, nb_rich=nb_rich,
                       fraction_endowment=fraction_endowment, homophily=homophily, risk=risk, M=M,
                       rich_endowment=rich_endowment, poor_endowment=poor_endowment, mu=mu, beta=beta,
                       rich_evolution=rich_evolution, poor_evolution=poor_evolution)

    #Game.play()
    #Game.GraphStationaryDistribution()
    Game.GraphOnePopEvolution(evolving_population="P", ratio_cooperators=[.1, .5, .9], rich_endowments=[1.35, 1.75])

    # print(len(Game.sample(0.8)[32]))

    # for i in range(6):
    # print(Game.sample(0.8)[32][i].get_wealth())

    # a = Game.sample(0.8)[32]
    # print(len(a))
    # print(Game.get_comp(a))

    # print(Game.payoffs_[:,32])

    # print("fitness:", Game.calculate_fitness(20, 80))
    # [0.5271957179851917, 2.108782871940767, 0.49202061373114003, 2.0751206672259306]

    # print(Game.calculate_payoffs())

    # print("Results[PC_to_PD, PD_to_PC, RC_to_RD, RD_to_RC]:", Game.transition_probabilities(ir=20, ip=60))
