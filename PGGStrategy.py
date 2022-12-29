"""
The PGGStrategy.py contains strategy which can be used by the PGG model.
"""

class PGGStrategy:

    def __init__(self, action: int) -> None:

        self.action = action

    def get_action(self) -> int:

        return self.action

    @property

    def type(self) -> str:

        return "PGGStrategy"