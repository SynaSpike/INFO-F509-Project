"""
The PGGStrategy.py contains strategy which can be used by the PGG model.
"""

class PGGStrategy:


    def init(self, strategy):
        self.strategy = strategy
        # TODO add rich/poor selection here
        self.wealth = "rich"

    def get_action(self, game_state):
        # TODO add mutation here ?
        if self.strategy == "cooperate":
            return 1
        elif self.strategy == "defect":
            return 0
        else:
            raise ValueError("Invalid strategy.")

    def type(self) -> str:
        return "PGGStrategy"