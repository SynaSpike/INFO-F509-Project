class Player:

    def __init__(self, wealth: int) -> None:

        self.wealth = wealth

    def get_wealth(self) -> int:

        return self.wealth

    @property
    def type(self) -> str:

        return "Player"