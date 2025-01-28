import random
from collections import UserDict

MAX_SCORE = 10
MIN_SCORE = 0

class Score:
    def __init__(self, score_values_in: list[float] = None, score_names_in: list[str] = None):       
        if score_values_in is None:
            score_values_in = []
        if score_names_in is None:
            self._score_names = self._default_score_names()
        else:
            self._score_names = score_values_in

        self._score_values = [random.uniform(MAX_SCORE, MIN_SCORE) for _ in range(len(self._score_names))]
        if len(score_values_in) <= len(self._score_names):
            for i, value in enumerate(score_values_in):
                self._score_values[i] = self._bounded_value(value)

        self._scores = BoundedUserDict(self._score_names, zip(self._score_names, self._score_values))

    @property
    def scores(self):
        return self._scores

    def _bounded_value(self, value: float):
        return max(min(value, MAX_SCORE), MIN_SCORE)
    
    @classmethod
    def _default_score_names(self):
        return [
                'english_proficiency',
                'clarity',
                'technical_depth',
                'aggressivity',
                'empathy'
            ]

    def get_max_score(self):
        return max(self._score_values)
    
    def get_min_score(self):
        return min(self._score_values)
    
class BoundedUserDict(UserDict):
    def __init__(self, score_names: list[str], *args, **kwargs):
        self.score_names = score_names
        super().__init__(*args, *kwargs)

    def __setitem__(self, key, value):
        if key not in self.score_names:
            print(f"Invalid score name to assign: {key}")
        bounded_value = max(MIN_SCORE, min(MAX_SCORE, value))
        super().__setitem__(key, bounded_value)
    
    def __json__(self):
        return dict(self)