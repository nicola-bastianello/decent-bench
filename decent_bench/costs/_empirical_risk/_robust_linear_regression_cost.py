from __future__ import annotations

from functools import cached_property

import numpy as np
from numpy import float64
from numpy.typing import NDArray

import decent_bench.utils.interoperability as iop
from decent_bench.costs._base._cost import Cost
from decent_bench.costs._base._sum_cost import SumCost
from decent_bench.costs._empirical_risk._empirical_risk_cost import EmpiricalRiskCost
from decent_bench.utils.types import (
    Dataset,
    EmpiricalRiskBatchSize,
    EmpiricalRiskIndices,
    EmpiricalRiskReduction,
    SupportedDevices,
    SupportedFrameworks,
)


class RobustLinearRegressionCost(EmpiricalRiskCost):
    r"""
    Robust linear regression cost function using the Huber loss.

    Given a data matrix :math:`\mathbf{A} \in \mathbb{R}^{m \times n}`, target vector
    :math:`\mathbf{b} \in \mathbb{R}^{m}`, and threshold :math:`\delta > 0`, the cost is:

    .. math::
        f(\mathbf{x}) = \frac{1}{m} \sum_{i=1}^m h_\delta(A_i x - b_i)

    where the Huber loss :math:`h_\delta` is defined per-sample as:

    .. math::
        h_\delta(r) =
        \begin{cases}
            \frac{1}{2} r^2 & \text{if} \ |r| \leq \delta, \\
            \delta \left(|r| - \frac{1}{2}\delta \right) & \text{if} \ |r| > \delta.
        \end{cases}

    In the stochastic setting, a mini-batch of size :math:`b < m` replaces the full dataset.
    """

    def __init__(self, dataset: Dataset, batch_size: EmpiricalRiskBatchSize = "all", delta: float = 1.0):
        """
        Initialize a RobustLinearRegressionCost instance.

        Args:
            dataset (Dataset): Dataset containing features and targets. The expected shapes are:
                - Features: (n_features,)
                - Targets: single dimensional values
            batch_size (EmpiricalRiskBatchSize): Size of mini-batches for stochastic methods, or "all" for full-batch.
            delta (float): Huber threshold separating the quadratic and linear regions. Must be positive.

        Raises:
            ValueError: If input dimensions are inconsistent, batch_size is invalid, or delta is nonpositive.
            TypeError: If dataset targets are not single dimensional values.

        """
        if len(iop.shape(dataset[0][0])) != 1:
            raise ValueError(f"Dataset features must be vectors, got: {dataset[0][0]}")
        if iop.to_numpy(dataset[0][1]).shape != (1,):
            raise TypeError(
                f"Dataset targets must be single dimensional values, got: {dataset[0][1]} "
                f"with shape {iop.to_numpy(dataset[0][1]).shape}, expected shape is (1,)."
            )
        if isinstance(batch_size, int) and (batch_size <= 0 or batch_size > len(dataset)):
            raise ValueError(
                f"Batch size must be positive and at most the number of samples, "
                f"got: {batch_size} and number of samples is: {len(dataset)}."
            )
        if isinstance(batch_size, str) and batch_size != "all":
            raise ValueError(f"Invalid batch size string. Supported value is 'all', got {batch_size}.")
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}.")

        self._dataset = dataset
        self._delta = delta
        self._batch_size = self.n_samples if batch_size == "all" else batch_size
        # Cache data matrices for efficiency when using full dataset
        self.A: NDArray[float64] | None = None
        self.b: NDArray[float64] | None = None
        self.ATA: NDArray[float64] | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return iop.shape(self._dataset[0][0])

    @property
    def framework(self) -> SupportedFrameworks:
        return SupportedFrameworks.NUMPY

    @property
    def device(self) -> SupportedDevices:
        return SupportedDevices.CPU

    @property
    def n_samples(self) -> int:
        return len(self._dataset)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @cached_property
    def m_smooth(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's smoothness constant:

        .. math::
            \frac{\lambda_{\max}(\mathbf{A}^T \mathbf{A})}{m}

        For the general definition, see
        :attr:`Cost.m_smooth <decent_bench.costs.Cost.m_smooth>`.
        """
        _, ATA, _ = self._get_batch_data(indices="all")  # noqa: N806
        eigs = np.linalg.eigvalsh(ATA)
        return float(np.max(np.abs(eigs))) / self.n_samples

    @cached_property
    def m_cvx(self) -> float:  # pyright: ignore[reportIncompatibleMethodOverride]
        r"""
        The cost function's strong convexity constant.

        The Huber loss is convex but not strongly convex in general: samples whose
        residual exceeds :math:`\delta` contribute zero curvature, so the global
        strong convexity constant is :math:`0`.

        For the general definition, see
        :attr:`Cost.m_cvx <decent_bench.costs.Cost.m_cvx>`.
        """
        return 0

    @iop.autodecorate_cost_method(EmpiricalRiskCost.predict)
    def predict(self, x: NDArray[float64], data: list[NDArray[float64]]) -> NDArray[float64]:
        r"""
        Make predictions at x on the given data.

        The predicted targets are computed as :math:`\mathbf{Ax}`.

        Args:
            x: Point to make predictions at.
            data: List of NDArray containing data to make predictions on.

        Returns:
            Predicted targets as an array.

        """
        pred_data = np.stack(data) if isinstance(data, list) else data
        pred: NDArray[float64] = pred_data.dot(x)
        return pred

    @staticmethod
    def _huber(residuals: NDArray[float64], delta: float) -> NDArray[float64]:
        """Per-sample Huber loss values."""
        abs_r = np.abs(residuals)
        return np.where(abs_r <= delta, 0.5 * residuals**2, delta * (abs_r - 0.5 * delta))

    @staticmethod
    def _huber_grad(residuals: NDArray[float64], delta: float) -> NDArray[float64]:
        """Pseudo-gradient of the Huber loss: clip(r, -delta, delta)."""
        return np.clip(residuals, -delta, delta)

    @iop.autodecorate_cost_method(EmpiricalRiskCost.function)
    def function(self, x: NDArray[float64], indices: EmpiricalRiskIndices = "batch") -> float:
        r"""
        Evaluate function at x using datapoints at the given indices.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        If no batching is used, this is:

        .. math::
            \frac{1}{m} \sum_{i=1}^m L_\delta(A_i x - b_i)

        If indices is "batch", a random batch :math:`\mathcal{B}` is drawn with :attr:`batch_size` samples.
        """
        A, _, b = self._get_batch_data(indices)  # noqa: N806
        residuals = A.dot(x) - b
        return float(np.sum(self._huber(residuals, self._delta)) / len(self.batch_used))

    @iop.autodecorate_cost_method(EmpiricalRiskCost.gradient)
    def gradient(
        self,
        x: NDArray[float64],
        indices: EmpiricalRiskIndices = "batch",
        reduction: EmpiricalRiskReduction = "mean",
    ) -> NDArray[float64]:
        r"""
        Gradient at x using datapoints at the given indices.

        Supported values for indices are:
            - int: datapoint to use.
            - list[int]: datapoints to use.
            - "all": use the full dataset.
            - "batch": draw a batch with :attr:`batch_size` samples.

        Supported values for reduction are:
            - "mean": average the gradients over the samples.
            - None: return the gradients for each sample, index as the first dimension.

        If no batching is used, this is:

        .. math::
            \frac{1}{m} \mathbf{A}^T \psi_\delta(\mathbf{Ax} - \mathbf{b})

        where :math:`\psi_\delta(r) = \operatorname{clip}(r, -\delta, \delta)`.

        Note:
            When reduction is None, the returned array will have an additional leading dimension
            corresponding to the number of samples used. Indexing into this dimension will give the gradient
            for the respective sample in :attr:`batch_used <decent_bench.costs.EmpiricalRiskCost.batch_used>`.

        """
        if reduction is None:
            return self._per_sample_gradients(x, indices)

        A, _, b = self._get_batch_data(indices)  # noqa: N806
        psi = self._huber_grad(A.dot(x) - b, self._delta)
        res: NDArray[float64] = A.T.dot(psi) / len(self.batch_used)
        return res

    def _per_sample_gradients(
        self,
        x: NDArray[float64],
        indices: EmpiricalRiskIndices = "batch",
    ) -> NDArray[float64]:
        A, _, b = self._get_batch_data(indices)  # noqa: N806
        psi = self._huber_grad(A.dot(x) - b, self._delta)  # shape: (n_samples,)
        res: NDArray[float64] = psi[:, np.newaxis] * A
        return res

    @iop.autodecorate_cost_method(EmpiricalRiskCost.hessian)
    def hessian(self, x: NDArray[float64], indices: EmpiricalRiskIndices = "batch") -> NDArray[float64]:
        r"""
        Generalized Hessian at x using datapoints at the given indices.

        The Huber loss is not twice differentiable everywhere; in the linear region
        (|residual| > delta) the second derivative is zero.  The generalized Hessian is:

        .. math::
            \frac{1}{m} \mathbf{A}^T \operatorname{diag}(\mathbf{d}) \mathbf{A},
            \quad d_i = \mathbf{1}[|A_i x - b_i| \leq \delta].

        Supported values for indices are the same as :meth:`function`.
        """
        A, _, b = self._get_batch_data(indices)  # noqa: N806
        residuals = A.dot(x) - b
        d = (np.abs(residuals) <= self._delta).astype(float64)
        AD = d[:, np.newaxis] * A  # scale rows of A by d
        res: NDArray[float64] = A.T.dot(AD) / len(self.batch_used)
        return res

    def _get_batch_data(
        self,
        indices: EmpiricalRiskIndices = "batch",
    ) -> tuple[NDArray[float64], NDArray[float64], NDArray[float64]]:
        """Get data for a batch. Returns A, A.T@A and b for the batch."""
        indices = self._sample_batch_indices(indices)

        if len(indices) == self.n_samples:
            if self.A is None or self.b is None or self.ATA is None:
                self.A = np.stack([iop.to_numpy(x) for x, _ in self._dataset])
                self.b = np.stack([iop.to_numpy(y) for _, y in self._dataset]).squeeze()
                self.ATA = self.A.T @ self.A
            return self.A, self.ATA, self.b

        A_list, b_list = [], []  # noqa: N806
        for idx in indices:
            x_i, y_i = self._dataset[idx]
            A_list.append(iop.to_numpy(x_i))
            b_list.append(iop.to_numpy(y_i))
        A = np.stack(A_list)  # noqa: N806
        b = np.stack(b_list).squeeze()
        return A, A.T @ A, b

    def __add__(self, other: Cost) -> Cost:
        """
        Add another cost function.

        When adding two :class:`RobustLinearRegressionCost` instances with the same
        ``delta``, the datasets are merged into a single cost for efficiency.

        Raises:
            ValueError: if the domain shapes don't match

        """
        if self.shape != other.shape:
            raise ValueError(f"Mismatching domain shapes: {self.shape} vs {other.shape}")
        if isinstance(other, RobustLinearRegressionCost) and self._delta == other._delta:
            if self.batch_size == self.n_samples and other.batch_size == other.n_samples:
                combined_batch_size: EmpiricalRiskBatchSize = self.n_samples + other.n_samples
            elif self.batch_size == self.n_samples:
                combined_batch_size = other.batch_size
            elif other.batch_size == other.n_samples:
                combined_batch_size = self.batch_size
            else:
                combined_batch_size = max(self.batch_size, other.batch_size)

            return RobustLinearRegressionCost(
                dataset=self.dataset + other.dataset,
                delta=self._delta,
                batch_size=combined_batch_size,
            )
        return SumCost([self, other])
