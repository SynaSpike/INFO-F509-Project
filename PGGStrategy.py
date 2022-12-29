"""
The PGGStrategy.py contains strategy which can be used by the PGG model.
"""

class PGGStrategy:

    def __init__(self, action: int) -> None:
        """
        :param action: 1 being cooperator and 0 being defector
        """

        self.action = action

    def get_action(self) -> int:

        return self.action

    @property

    def type(self) -> str:

        return "PGGStrategy"

    def __str__(self):
        return "PGGStrategy Object"