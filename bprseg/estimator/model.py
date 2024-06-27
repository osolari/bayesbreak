from sklearn.base import BaseEstimator, RegressorMixin, _fit_context


class ProcessSegmentation(RegressorMixin, BaseEstimator):

    def __init__(self, num_jobs: int = 1) -> None:
        super().__init__()

        self.num_jobs = num_jobs

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """_summary_

        Args:
            X (_type_): _description_
            y (_type_): _description_
            sample_weight (_type_, optional): _description_. Defaults to None.
        """

        pass
