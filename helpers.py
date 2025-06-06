from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import BaseCrossValidator

class RepeatedStratifiedGroupKFold(BaseCrossValidator):
    """
    Repeated Stratified *Group* K-Fold cross-validator.

    Parameters
    ----------
    n_splits  : int, default=5
        Number of folds in each repeat.
    n_repeats : int, default=10
        How many times to re-draw the folds.
    random_state : int or None, default=None
        Seed used to generate a new random_state for every repeat.
    """

    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y, groups):
        n_samples = _num_samples(X)
        if groups is None:
            raise ValueError("The 'groups' parameter must not be None.")

        rng_master = check_random_state(self.random_state)

        for rep in range(self.n_repeats):
            # fresh RNG for this repeat → independent shuffle
            rng_repeat = check_random_state(rng_master.randint(0, 2**32 - 1))

            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=rng_repeat,
            )
            yield from cv.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits * self.n_repeats

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(n_splits={self.n_splits}, "
            f"n_repeats={self.n_repeats}, random_state={self.random_state})"
        )
