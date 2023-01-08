""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""
import random
import matplotlib.colors
import numpy as np
import math
import matplotlib.pyplot as plt
from typing import Union, Any
from scipy.special import comb as comb
from scipy.linalg import eig as eig


class ClimateGame:

    def __init__(self, popuplation_size: int, group_size: int, nb_rich: int, fraction_endowment: float,
                 homophily: float,
                 risk: float, M: float, rich_endowment: int, poor_endowment: int, mu: float, beta: float) -> None:

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
        self.M = M
        self.threshold = M * fraction_endowment * self.b_bar
        self.threshold_uncertainty = 0
        self.mu = mu
        self.beta = beta

        self.rich_evolution = 1
        self.poor_evolution = 1

        self.obstinate_players_ratio = False

    def update_endowments(self, rich_endowment):
        self.rich_endowment = rich_endowment
        poor_endowment = (self.population_size - (rich_endowment * self.rich))/self.poor
        self.poor_endowment = poor_endowment
        self.poor_contribution = poor_endowment * fraction_endowment  # Contribution of the Poor
        self.rich_contribution = rich_endowment * fraction_endowment  # Contribution of the rich
        self.b_bar = ((poor_endowment * self.poor) + (rich_endowment * self.rich)) / self.population_size
        self.threshold = M * fraction_endowment * self.b_bar
        print("Endowments update: br="+str(self.rich_endowment)+"; bp="+str(self.poor_endowment))

    def get_p(self):
        print("Calculating eigen values...")
        eigs, leftv, rightv = eig(self.W, left=True, right=True)
        print("Getting the index of the dominant eigenvalue...")
        domIdx = np.argmax(np.real(eigs))
        print("Getting the dominant eigenvalue...")
        L = np.real(eigs[domIdx])
        print("Getting the right-eigenvector...")
        p = np.real(rightv[:, domIdx])
        print("pmax =", max(p))
        print("Normalising the relative proportions...")

        p = p / np.sum(p)
        self.p = p
        print("pmax_norm =", max(p))

    def get_ng(self):
        print("Calculating ng...")
        ng = 0
        for index, P_bar_i in enumerate(self.p):
            ir, ip = self.populations_configurations[index]
            ng += P_bar_i * self.calculate_ag(ir, ip)

        self.ng = ng

        print("ng:", ng)

    def play(self):
        populations_configurations = []
        populations_configurations_index = {}
        index = 0
        print("Setting up all possible population configuration...")
        for ip in range(self.poor + 1):
            for ir in range(self.rich + 1):
                if self.obstinate_players_ratio:
                    if ir < self.obstinate_players["RC"]:
                        continue
                    if self.rich - ir < self.obstinate_players["RD"]:
                        continue
                    if ip < self.obstinate_players["PC"]:
                        continue
                    if self.poor - ip < self.obstinate_players["PD"]:
                        continue
                populations_configurations.append((ir, ip))
                populations_configurations_index[(ir, ip)] = index
                index += 1

        self.W = np.zeros((index, index))
        self.populations_configurations = populations_configurations
        self.populations_configurations_index = populations_configurations_index
        self.populations_transitions_results = {}
        self.totalindex = index

        print("Calculating population transitions...")
        for index, pop_config in enumerate(self.populations_configurations):
            self.calculate_population_transitions(pop_config)

        self.get_p()
        self.get_ng()

        # result[index]
        # 0 == (kX = rich defect -> kY = rich coop)
        # 1 == (kX = rich coop -> kY = rich defect)
        # 2 == (kX = poor defect -> kY = poor coop)
        # 3 == (kX = poor coop -> kY = poor defect)
        # 4 == (no transi)

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

        self.play()

        ZP = self.poor
        ZR = self.rich
        iV = self.populations_configurations
        grad_iR = self.gradient_rich
        grad_iP = self.gradient_poor

        P = np.zeros((ZP + 1, ZR + 1))  # rich on the x-axis
        for idx, pi in enumerate(self.p):
            iR, iP = iV[idx]
            P[iP, iR] = pi

        fig, ax = plt.subplots(figsize=(3, 6))

        x=[]
        for ip in range(ZP + 1):
            for ir in range(ZR + 1):
                x.append(ir)
        y=[]
        for ip in range(ZP + 1):
            for ir in range(ZR + 1):
                y.append(ip)

        customcmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#DCDCDC", "black"])
        plt.scatter(x, y, c=P, alpha=0.85, cmap=customcmap, edgecolors="#A9A9A9")

        iRV = list(range(ZR + 1))
        iPV = list(range(ZP + 1))

        grad_ir_array = np.zeros((ZP + 1, ZR + 1))
        grad_ip_array = np.zeros((ZP + 1, ZR + 1))
        colors = np.zeros((ZP + 1, ZR + 1))

        for index, gradient in enumerate(grad_iR):
            ir, ip = self.populations_configurations[index]
            grad_ir_array[ip][ir] = gradient
            colors[ip][ir] += abs(gradient)

        for index, gradient in enumerate(grad_iP):
            ir, ip = self.populations_configurations[index]
            grad_ip_array[ip][ir] = gradient
            colors[ip][ir] += abs(gradient)

        customcolormap2 = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#610484", "#5a2293",
        "#5340a1", "#366695", "#128f81", "#22a967", "#67b448", "#9fae31", "#b17630", "#c33d30"])
        ax.streamplot(iRV, iPV, grad_ir_array, grad_ip_array, color=colors, density=.5, cmap=customcolormap2)

        # sm = matplotlib.cm.ScalarMappable(cmap=customcmap)
        # sm.set_array([])
        #
        # cbar1 = plt.colorbar(sm, ax=ax)
        # cbar1.ax.get_yaxis().set_ticks([])
        # cbar1.ax.get_yaxis().labelpad = 15
        # cbar1.ax.text(1.1, 0, "$0$")
        # cbar1.ax.text(1.1, 0.975, "$p^{max}_k$")
        #
        # sm = matplotlib.cm.ScalarMappable(cmap=customcolormap2)
        # sm.set_array([])
        #
        # cbar2 = plt.colorbar(sm, ax=ax)
        # cbar2.ax.get_yaxis().set_ticks([])
        # cbar2.ax.get_yaxis().labelpad = 15
        # cbar2.ax.text(1.1, 0, "$0$")
        # cbar2.ax.text(1.1, 0.975, "$∇^{max}_i$")

        ax.set_xlim((-1, ZR + 1))
        ax.set_ylim((-1, ZP + 1))
        ax.set_xlabel(r'rich cooperators, $i_R$')
        ax.set_ylabel(r'poor cooperators, $i_P$')
        plt.axis('scaled')
        plt.tight_layout()
        plt.show()

    def GraphStationaryDistribution_obstinate(self, obstinate_players_ratio:list):

        self.obstinate_players_ratio = obstinate_players_ratio

        self.obstinate_players = {
            "RC":int(self.rich * obstinate_players_ratio[0]), #obstinate rich coop
            "RD":int(self.rich * obstinate_players_ratio[1]), #obstinate rich defect
            "PC":int(self.poor * obstinate_players_ratio[2]), #obstinate poor coop
            "PD":int(self.poor * obstinate_players_ratio[3]), #obstinate poor defect
        }

        ZP = self.poor
        ZR = self.rich

        self.play()

        iV = self.populations_configurations
        grad_iR = self.gradient_rich
        grad_iP = self.gradient_poor

        P = np.zeros((ZP + 1, ZR + 1))  # rich on the x-axis
        for idx, pi in enumerate(self.p):
            iR, iP = iV[idx]
            P[iP, iR] = pi

        fig, ax = plt.subplots(figsize=(3, 6))

        x=[]
        for ip in range(ZP + 1):
            for ir in range(ZR + 1):
                x.append(ir)
        y=[]
        for ip in range(ZP + 1):
            for ir in range(ZR + 1):
                y.append(ip)

        customcmap = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#DCDCDC", "black"])
        plt.scatter(x, y, c=P, alpha=0.85, cmap=customcmap, edgecolors="#A9A9A9")

        iRV = list(range(ZR + 1))
        iPV = list(range(ZP + 1))

        grad_ir_array = np.zeros((ZP + 1, ZR + 1))
        grad_ip_array = np.zeros((ZP + 1, ZR + 1))
        colors = np.zeros((ZP + 1, ZR + 1))

        for index, gradient in enumerate(grad_iR):
            ir, ip = self.populations_configurations[index]
            grad_ir_array[ip][ir] = gradient
            colors[ip][ir] += abs(gradient)

        for index, gradient in enumerate(grad_iP):
            ir, ip = self.populations_configurations[index]
            grad_ip_array[ip][ir] = gradient
            colors[ip][ir] += abs(gradient)

        customcolormap2 = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#610484", "#5a2293",
        "#5340a1", "#366695", "#128f81", "#22a967", "#67b448", "#9fae31", "#b17630", "#c33d30"])
        ax.streamplot(iRV, iPV, grad_ir_array, grad_ip_array, color=colors, density=.5, cmap=customcolormap2)

        ax.set_xlim((-1, ZR + 1))
        ax.set_ylim((-1, ZP + 1))
        ax.set_xlabel(r'rich cooperators, $i_R$')
        ax.set_ylabel(r'poor cooperators, $i_P$')
        plt.axis('scaled')
        plt.tight_layout()
        plt.show()

    def GraphOnePopEvolution(self, evolving_population:str, ratio_cooperators:list, rich_endowments:list):

        evolv_pop_size = evolving_population == "R" and self.rich or self.poor
        self.rich_evolution = evolving_population == "R" and 1 or 0
        self.poor_evolution = 1 - self.rich_evolution

        fig = plt.figure()
        ax = plt.subplot(111)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        colors = ["blue", "orange", "green", "red", "purple"]

        x = []
        other_pop_size = self.population_size - evolv_pop_size

        for evolv_pop_coop in range(evolv_pop_size):
            x.append(evolv_pop_coop/evolv_pop_size)

        plt.plot(x, [0 for i in range(evolv_pop_size)], '-', color="black", linewidth=0.5)

        for rich_endowment in rich_endowments:
            self.update_endowments(rich_endowment=rich_endowment)

            self.play()

            evolv_pop_grad = evolving_population == "R" and self.gradient_rich or self.gradient_poor

            for index, ratio in enumerate(ratio_cooperators):
                other_pop_coop = int(other_pop_size * ratio)
                y = []
                for evolv_pop_coop in range(evolv_pop_size):
                    pop_config = evolving_population == "R" and (evolv_pop_coop, other_pop_coop) or (other_pop_coop, evolv_pop_coop)
                    index_ = self.populations_configurations_index[pop_config]
                    y.append(evolv_pop_grad[index_])

                if rich_endowment == rich_endowments[0]:
                    labeltext = str(int(ratio * 100))+"%"
                    plt.plot(x, y, '--', label=labeltext, color=colors[index])
                else:
                    plt.plot(x, y, '-', color=colors[index])

        plt.plot([], [], '--', label="$b_R$ > $b_P$", color="black")
        plt.plot([], [], '-', label="$b_R$ >> $b_P$", color="black")
        plt.legend(title="fraction of $C_P$", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
        popu = evolving_population == "R" and "rich" or "poor"
        ax.set_xlabel(r'fraction of '+popu+' cooperators, $i_'+evolving_population+'/Z_'+evolving_population+'$')
        ax.set_ylabel(r'gradient of selection, $∇_i$('+popu+')')
        plt.show()

    def GraphThresholdUncertainty(self, threshold_uncertainties):

        self.update_endowments(rich_endowment=1)
        self.wealth_inequality = True

        colors = ["black", "red"]

        x = []
        for i in range(self.population_size + 1):
            x.append(i/self.population_size)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax2.plot(x, [0 for i in x], '-', color="black", linewidth=0.5)

        for color_index, threshold_uncertainty in enumerate(threshold_uncertainties):
            self.threshold_uncertainty = threshold_uncertainty
            self.play()

            stat_dist = [[] for i in range(self.population_size + 1)]
            grad_sel = [[] for i in range(self.population_size + 1)]

            for index, config in enumerate(self.populations_configurations):
                ir, ip = config
                nbr_coop = ir+ip
                stat_dist[nbr_coop].append(self.p[index])
                grad_sel[nbr_coop].append(self.gradient_selection[index])

            stat_dist_avg = [0 for i in range(self.population_size + 1)]
            grad_sel_avg = [0 for i in range(self.population_size + 1)]

            for nbr_coop, dist in enumerate(stat_dist):
                stat_dist_avg[nbr_coop] = max(dist)

            for nbr_coop, gradients in enumerate(grad_sel):
                total = 0

                for gradient in gradients:
                    total += (gradient[0]+gradient[1])/2

                grad_sel_avg[nbr_coop] = total / len(gradients)

            labeltext1 = "δ = "+str(threshold_uncertainty)
            ax1.plot(x, stat_dist_avg, color_index == 0 and '-' or "--", label=labeltext1, color=colors[color_index])
            labeltext2 = "δ = "+str(threshold_uncertainty)
            ax2.plot(x, grad_sel_avg, color_index == 0 and '-' or "--", label=labeltext2, color=colors[color_index])

        ax1.legend(loc='upper right', frameon=False)
        ax2.legend(loc='upper right', frameon=False)
        plt.show()

    def Graph_ng_risk(self, homophilies, wealth_inequalites, threshold_uncertainties, Ns, Ms):
        population_rich = self.rich
        rich_endowment = self.rich_endowment

        colors = ["blue", "red", "grey"]

        fig = plt.figure()
        ax = plt.subplot(111)

        x = []
        for i in range(0, 101, 5):
            x.append(i/100)

        for i in range(len(homophilies)):
            self.homophily = homophilies[i]
            self.threshold_uncertainty = threshold_uncertainties[i]
            self.N = Ns[i]
            self.M = Ms[i]
            wealth_inequality = wealth_inequalites[i]

            if wealth_inequality:
                self.rich = int(population_rich)
                self.poor = int(self.population_size-population_rich)
                self.update_endowments(rich_endowment=rich_endowment)
            else:
                self.rich = int(self.population_size/2)
                self.poor = int(self.rich)
                self.update_endowments(rich_endowment=1)

            y = []

            for risk in range(0, 101, 5):
                risk = risk/100
                self.risk = risk

                self.play()

                y.append(self.ng)

            #labeltext = "h="+str(self.homophily)+" ;δ="+str(self.threshold_uncertainty)+"; w_inequality:"+str(wealth_inequality)
            plt.plot(x, y, '-', color=colors[i])

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        #plt.legend(loc='best')
        ax.set_xlabel(r'risk, $r$')
        ax.set_ylabel(r'group achievement, $η_G$')
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
        jp = group_composition[2]
        jr = group_composition[3]

        game_payoffs = np.zeros(self.nb_strategies)

        threshold_value = random.uniform(self.threshold - self.threshold_uncertainty, self.threshold + self.threshold_uncertainty)

        k = self.rich_contribution * jr + self.poor_contribution * jp - threshold_value

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
                if self.obstinate_players_ratio:
                    if ir_prime < self.obstinate_players["RC"]:
                        transitions_results[len(transitions_results)] = 0
                        continue
                    if self.rich - ir_prime < self.obstinate_players["RD"]:
                        transitions_results[len(transitions_results)] = 0
                        continue
                    if ip_prime < self.obstinate_players["PC"]:
                        transitions_results[len(transitions_results)] = 0
                        continue
                    if self.poor - ip_prime < self.obstinate_players["PD"]:
                        transitions_results[len(transitions_results)] = 0
                        continue

                transition_index = self.populations_configurations_index[(ir_prime, ip_prime)]

                Zr = self.rich
                Zp = self.poor

                if transition == (1, -1, 0, 0):
                    # print("")
                    # print("Transition kX = rich defect -> kY = rich coop")
                    # k = rich, l = poor, X = defect, Y = coop
                    result = self.rich_evolution * self.transition_probability(Zk=Zr, Zl=Zp,
                                                                               ikX=max(Zr - ir, 0), ikY=ir, ilY=ip,
                                                                               fkX=fitness[1], fkY=fitness[3],
                                                                               flY=fitness[2])

                elif transition == (-1, 1, 0, 0):
                    # print("Transition kX = rich coop -> kY = rich defect")
                    # k = rich, l = poor, X = coop, Y = defect
                    result = self.rich_evolution * self.transition_probability(Zk=Zr, Zl=Zp, ikX=ir,
                                                                               ikY=max(Zr - ir, 0), ilY=max(Zp - ip, 0),
                                                                               fkX=fitness[3], fkY=fitness[1],
                                                                               flY=fitness[0])

                elif transition == (0, 0, 1, -1):
                    # print("Transition kX = poor defect -> kY = poor coop")
                    # k = pauvre, l = riche, X = defect, Y = coop
                    result = self.poor_evolution * self.transition_probability(Zk=Zp, Zl=Zr,
                                                                               ikX=max(Zp - ip, 0), ikY=max(ip, 0), ilY=ir,
                                                                               fkX=fitness[0], fkY=fitness[2],
                                                                               flY=fitness[3])

                elif transition == (0, 0, -1, 1):
                    # print("Transition kX = poor coop -> kY = poor defect")
                    # k = pauvre, l = riche, X = coop, Y = defect
                    result = self.poor_evolution * self.transition_probability(Zk=Zp, Zl=Zr, ikX=ip,
                                                                               ikY=max(Zp - ip, 0), ilY=max(Zr - ir, 0),
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
        Z = self.population_size
        N = self.group_size
        # Multivariate hypergeometric sampling (fitness equations) to compute the (average) fraction of groups that
        # reach a total of Mcb in contributions
        return sum(comb(iR, jR) * comb(iP, jP) * comb(Z - iR - iP, N - jR - jP) * self.contribution_reached(jR, jP)
                   for jR in range(N + 1) for jP in range(N + 1)) / comb(Z, N)

    @staticmethod
    def __str__(self) -> str:
        """
        This method should return a string representation of the game.
        """
        return "ClimateGame Object"


if __name__ == '__main__':
    population_size = 25
    nb_rich = 10
    group_size = 8
    rich_endowment = 1
    poor_endowment = 1

    fraction_endowment = 0.1
    homophily = 0
    risk = 0.6
    M = 4  # Between 0 and group_size

    mu = 1/population_size
    beta = 6

    Game = ClimateGame(popuplation_size=population_size, group_size=group_size, nb_rich=nb_rich,
                       fraction_endowment=fraction_endowment, homophily=homophily, risk=risk, M=M,
                       rich_endowment=rich_endowment, poor_endowment=poor_endowment, mu=mu, beta=beta)

    # Game.GraphStationaryDistribution()
    # Game.GraphOnePopEvolution(evolving_population="R", ratio_cooperators=[.1, .5, .9], rich_endowments=[1.35, 1.75])
    # Game.GraphThresholdUncertainty(threshold_uncertainties=[0, 2.75])
    # Game.GraphStationaryDistribution_obstinate(obstinate_players_ratio=[0, 0, 0.1, 0])
    # Game.Graph_ng_risk(homophilies=[0, 1, 0], wealth_inequalites=[True, True, False], threshold_uncertainties=[0, 0, 0],
    #                    Ns=[6, 6, 6], Ms=[3, 3, 3])


