class Player:

    def __init__(self, wealth: int, endowment: int) -> None:
        """
        :param wealth: 1 being rich and 0 being poor.
        :param: endowment
        """

        self.wealth = wealth
        self.endowment = endowment

    def get_wealth(self) -> int:

        return self.wealth

    @property
    def type(self) -> str:

        return "Player"

    def __str__(self):

        return "Player Object"