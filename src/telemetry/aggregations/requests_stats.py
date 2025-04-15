from typing import List, Dict
from dataclasses import dataclass
from collections import Counter

import numpy as np

from numba import jit

from telemetry.models import RequestResult


__all__ = ["RequestStats", "aggr_requests_stats"]


@dataclass
class PercentileStats:
    p25: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    @classmethod
    def default(cls) -> "PercentileStats":
        return cls(
            p25=0,
            p50=0,
            p75=0,
            p90=0,
            p95=0,
            p99=0
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99
        }


@dataclass
class HistogramData:
    bins: List[float]
    counts: List[int]

    @classmethod
    def default(cls) -> "HistogramData":
        return cls(
            bins=[],
            counts=[]
        )

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "bins": self.bins,
            "counts": self.counts
        }


@dataclass
class MovingAverageData:
    window_size: int
    values: List[float]
    timestamps: List[float]

    def to_dict(self) -> Dict[str, List[float]]:
        return {
            "window_size": self.window_size,
            "values": self.values,
            "timestamps": self.timestamps
        }


@dataclass
class RequestStats:
    avg: float
    min: float
    max: float
    total: float
    count: int
    std_dev: float
    variance: float
    percentiles: PercentileStats
    status_counts: Dict[str, int]
    error_counts: Dict[str, int]
    histogram: HistogramData
    throughput: float
    moving_averages: List[MovingAverageData]

    @classmethod
    def default(cls) -> "RequestStats":
        return cls(
            avg=0,
            min=0,
            max=0,
            total=0,
            count=0,
            std_dev=0,
            variance=0,
            percentiles=PercentileStats.default(),
            status_counts={},
            error_counts={},
            histogram=HistogramData.default(),
            throughput=0,
            moving_averages=[]
        )

    def to_dict(self) -> Dict[str, any]:
        return {
            "avg": self.avg,
            "min": self.min,
            "max": self.max,
            "total": self.total,
            "count": self.count,
            "std_dev": self.std_dev,
            "variance": self.variance,
            "percentiles": self.percentiles.to_dict(),
            "status_counts": self.status_counts,
            "error_counts": self.error_counts,
            "histogram": self.histogram.to_dict(),
            "throughput": self.throughput,
            "moving_averages": [ma.to_dict() for ma in self.moving_averages]
        }


@jit(nopython=True)
def calculate_moving_averages_numba(durations, timestamps, window_size):
    """
    Calculate moving averages using Numba for performance optimization.

    Args:
        durations: NumPy array of duration values
        timestamps: NumPy array of timestamp values
        window_size: Size of the moving window

    Returns:
        Tuple of (sma_values, sma_timestamps)
    """
    assert window_size != 0, "Window size cannot be zero"

    n = len(durations)
    result_size = n - window_size + 1

    # Pre-allocate arrays for results
    sma_values = np.zeros(result_size)
    sma_timestamps = np.zeros(result_size)

    # Calculate moving averages
    for i in range(result_size):
        # Calculate mean for this window
        window_sum = 0.0
        for j in range(window_size):
            window_sum += durations[i + j]
        sma_values[i] = window_sum / window_size

        # Use the last timestamp in the window
        sma_timestamps[i] = timestamps[i + window_size - 1]

    return sma_values, sma_timestamps


def aggr_requests_stats(requests: List[RequestResult]) -> RequestStats:
    """
    Calculate comprehensive statistics for a list of request results.

    This function analyzes request duration data and computes various statistical metrics
    including basic statistics (mean, min, max), distribution metrics (percentiles, histogram),
    status and error counts, throughput, and time-based moving averages.

    Args:
        requests: A list of RequestResult objects containing duration and timestamp data
                 for each request. Can be empty.

    Returns:
        RequestStats: A dataclass containing the following statistics:
            - avg: Average request duration
            - min: Minimum request duration
            - max: Maximum request duration
            - total: Sum of all request durations
            - count: Number of requests
            - median: Median request duration
            - std_dev: Standard deviation of request durations
            - variance: Variance of request durations
            - percentiles: Various percentile values (25th, 50th, 75th, 90th, 95th, 99th)
            - status_counts: Counter of request status values
            - error_counts: Counter of error messages
            - histogram: Binned distribution of request durations
            - throughput: Requests per second over the entire time period
            - moving_averages: Time-series moving averages with different window sizes

    Note:
        - Returns default values if the input list is empty
        - Uses NumPy for efficient statistical calculations
        - Calculates moving averages for window sizes of 5, 10, and 20 requests
        - Timestamps in moving averages correspond to the last request in each window
    """
    if not requests:
        return RequestStats.default()

    durations = np.array([r.duration_seconds for r in requests])

    # Calculate histogram data
    hist_counts, hist_bins = np.histogram(durations, bins='auto')

    # Count statuses
    status_counts = Counter([r.status.value for r in requests])

    # Count error types (if available)
    error_counts = Counter([r.error_message for r in requests if r.error_message])

    # Calculate moving averages
    moving_averages = []
    window_sizes = [20]

    sorted_requests = sorted(requests, key=lambda r: r.ts_created)
    sorted_durations = [r.duration_seconds for r in sorted_requests]
    timestamps = [r.ts_created for r in sorted_requests]

    # Convert to NumPy arrays once (outside the loop)
    np_sorted_durations = np.array(sorted_durations, dtype=np.float64)
    np_timestamps = np.array(timestamps, dtype=np.float64)

    for window in window_sizes:
        if len(sorted_durations) >= window:
            sma_values, sma_timestamps = calculate_moving_averages_numba(
                np_sorted_durations, np_timestamps, window
            )

            # Convert back to lists
            sma_values = sma_values.tolist()
            sma_timestamps = sma_timestamps.tolist()

            moving_averages.append(MovingAverageData(
                window_size=window,
                values=sma_values,
                timestamps=sma_timestamps
            ))

    # Calculate throughput if we have more than one request
    throughput = 0
    if len(requests) > 1:
        start_time = min(r.ts_created for r in requests)
        end_time = max(r.ts_created + r.duration_seconds for r in requests)
        duration = end_time - start_time
        if duration > 0:
            throughput = len(requests) / duration

    return RequestStats(
        avg=float(np.mean(durations)),
        min=float(np.min(durations)),
        max=float(np.max(durations)),
        total=float(np.sum(durations)),
        count=len(durations),
        std_dev=float(np.std(durations)),
        variance=float(np.var(durations)),
        percentiles=PercentileStats(
            p25=float(np.percentile(durations, 25)),
            p50=float(np.percentile(durations, 50)),
            p75=float(np.percentile(durations, 75)),
            p90=float(np.percentile(durations, 90)),
            p95=float(np.percentile(durations, 95)),
            p99=float(np.percentile(durations, 99))
        ),
        status_counts={k: v for k, v in status_counts.items()},
        error_counts={k: v for k, v in error_counts.items()},
        histogram=HistogramData(
            bins=hist_bins.tolist(),
            counts=hist_counts.tolist()
        ),
        throughput=throughput,
        moving_averages=moving_averages
    )
