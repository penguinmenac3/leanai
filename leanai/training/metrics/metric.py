"""doc
# leanai.training.metrics.metric

> An implementation of a metric.
"""
from leanai.training.losses.loss import Loss


class Metric(Loss):
    """
    You can either return a value that will be logged automatically
    or log the value yourself using self.log and return None.

    If you compute a single value returning is recommended,
    but if your metric consists of multiple numbers log them yourself.

    ```
    def forward(self, y_pred, y_true):
        self.log(...)
        return None

    def forward(self, y_pred, y_true):
        return 42
    ```
    """
    pass
