""""
This file is used to code for the main model of the article: https://www.pnas.org/doi/full/10.1073/pnas.1323479111
Vasconcelos VV, Santos FC, Pacheco JM, Levin SA. 2014
Climate policies under wealth inequality. Proc. Natl Acad. Sci. USA 111, 2212-2216.
(doi:10.1073/pnas.1323479111) Crossref, PubMed, ISI, Google Scholar
"""

from egttools import AbstractNPlayerGame

class PGG(AbstractNPlayerGame):

    def __init__(self, group_size: int, nb_rich:int, fraction_endowment: float, strategies: List[PGGstrat]):

        AbstractNPlayerGame.__init__(self, len(strategies), group_size)
        self.group_size_ = group_size  # Z
        self.Zr = nb_rich  # Zr
        self.Zp = group_size - nb_rich  # Zp
        self.fraction_endowment = fraction_endowment  # c
        self.strategies = ['Defect', 'Cooperate']



