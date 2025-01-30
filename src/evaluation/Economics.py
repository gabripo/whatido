class Employee:
    _name: str
    _productivity_scores: list[float] = []
    _profitability: float = float(0)

    def __init__(self, employee_name: str):
        self._name = employee_name

    def compute_profitability(self, productivity_scores: list[float]) -> None:
        # TODO implement
        self._productivity_scores = [1.0, 2.0]
        self._profitability = 1.0
        print(f"Employee {self._name} having scores {self._productivity_scores} has profitability {self._profitability}")
        pass

    def get_name(self) -> str:
        return self._name

    def get_profitability(self) -> float:
        return self._profitability