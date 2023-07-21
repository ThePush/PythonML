class TinyStatistician:
    def __init__(self):
        pass

    def mean(self, x):
        """Computes the mean of a given non-empty list or array x, using a for-loop.
        The method returns the mean as a float, otherwise None if x is an empty list or
        array."""
        if len(x) == 0:
            return None
        return sum(x) / len(x)

    def median(self, x):
        """Computes the median of a given non-empty list or array x. The method
        returns the median as a float, otherwise None if x is an empty list or array."""
        if len(x) == 0:
            return None
        x.sort()
        if len(x) % 2 == 0:
            return (float(x[int(len(x) / 2)]) + float(x[int(len(x) / 2) - 1])) / 2
        else:
            return float(x[int(len(x) / 2)])

    def quartile(self, x):
        """Computes the 1st and 3rd quartiles of a given non-empty array x.
        The method returns the quartile as a float, otherwise None if x is an empty list or
        array."""
        x.sort()
        return [self.median(x[1 : len(x) // 2]), self.median(x[len(x) // 2 :])]

    def percentile(self, x, p):
        """Computes the pth percentile of a given non-empty array x.
        The method returns the percentile as a float, otherwise None if x is an empty list or
        array."""
        if not x:
            return None
        if not 0 <= p <= 100:
            return None
        sorted_x = sorted(x)
        index = (p / 100) * (len(sorted_x) - 1)
        lower_index = int(index)
        upper_index = lower_index + 1
        lower_value = sorted_x[lower_index]
        upper_value = sorted_x[upper_index]
        fractional_part = index - lower_index
        percentile_value = (
            1 - fractional_part
        ) * lower_value + fractional_part * upper_value
        return percentile_value

    def var(self, x):
        """Computes the variance of a given non-empty list or array x, using a for-
        loop. The method returns the variance as a float, otherwise None if x is
        an empty list or array."""
        if len(x) == 0:
            return None
        mean = self.mean(x)
        return sum([(i - mean) ** 2 for i in x]) / len(x)

    def std(self, x):
        """Computes the standard deviation of a given non-empty list or array x,
        using a for-loop. The method returns the standard deviation as a float, otherwise
        None if x is an empty list or array."""
        if len(x) == 0:
            return None
        return self.var(x) ** 0.5


if __name__ == "__main__":
    a = [1, 42, 300, 10, 59]
    print(f"Test Mean: Expected: 82.4, Actual: {TinyStatistician().mean(a)}")
    print(f"Test Median: Expected: 42.0, Actual: {TinyStatistician().median(a)}")
    print(
        f"Test Quartile: Expected: [10.0, 59.0], Actual: {TinyStatistician().quartile(a)}"
    )
    print(
        f"Test Percentile(a, 10): Expected: 4.6, Actual: {TinyStatistician().percentile(a, 10)}"
    )
    print(
        f"Test Percentile(a, 15): Expected: 6.4, Actual: {TinyStatistician().percentile(a, 15)}"
    )
    print(
        f"Test Percentile(a, 20): Expected: 8.2, Actual: {TinyStatistician().percentile(a, 20)}"
    )
    print(f"Test Variance: Expected: 15349.3, Actual: {TinyStatistician().var(a)}")
    print(
        f"Test Standard Deviation: Expected: 123.89229193133849, Actual: {TinyStatistician().std(a)}"
    )
