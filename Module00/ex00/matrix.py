from dataclasses import dataclass, field
from numbers import Number


@dataclass
class Matrix:
    data: list = field(default_factory=list)
    shape: tuple = field(default_factory=tuple)

    def __init__(self, values):
        if isinstance(values, list):
            if not all(isinstance(x, list) for x in values) or not all(
                isinstance(y, Number) for x in values for y in x
            ):
                raise ValueError("Matrix invalid list type")
            if len(values) > 1:
                for row in values:
                    if len(row) != len(values[0]):
                        raise ValueError("Matrix invalid list size")
            self.data = values
            self.shape = (len(self.data), len(self.data[0]))
        elif isinstance(values, tuple):
            if len(values) != 2 or not all(isinstance(x, int) for x in values):
                raise ValueError("Matrix invalid tuple")
            self.shape = values
            self.data = [
                [0.0 for _ in range(self.shape[1])] for _ in range(self.shape[0])
            ]
        else:
            raise ValueError("Matrix invalid type")

    def __add__(self, other):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Matrix invalid type or shape")
        result = Matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("Matrix invalid type or shape")
        result = Matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[i][j] = self.data[i][j] - other.data[i][j]
        return result

    def __rsub__(self, other):
        return other.__sub__(self)

    def __truediv__(self, scalar: Number):
        if not isinstance(scalar, Number):
            raise ValueError("Matrix invalid type")
        result = Matrix(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[i][j] = self.data[i][j] / scalar
        return result

    def __rtruediv__(self, scalar: Number):
        return self.__truediv__(1 / scalar)

    def __mul__(self, other):
        if isinstance(other, Number):
            result = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][j] = self.data[i][j] * other
            return result
        elif isinstance(other, Vector):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix invalid shape")
            result = Vector([[0] for _ in range(self.shape[0])])
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][0] += self.data[i][j] * other.data[j][0]
            return result
        elif isinstance(other, Matrix):
            if self.shape[1] != other.shape[0]:
                raise ValueError("Matrix invalid shape")
            result = Matrix((self.shape[0], other.shape[1]))
            for i in range(self.shape[0]):
                for j in range(other.shape[1]):
                    for k in range(self.shape[1]):
                        result.data[i][j] += self.data[i][k] * other.data[k][j]
            return result
        else:
            raise ValueError("Matrix invalid type")

    def __rmul__(self, other):
        return self.__mul__(other)

    def T(self):
        result = Matrix((self.shape[1], self.shape[0]))
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                result.data[j][i] = self.data[i][j]
        return result

    def __str__(self):
        return f"{type(self)}: {str(self.data)}"

    def __repr__(self):
        return f"{type(self)}: {str(self.data)}"


class Vector(Matrix):
    def __init__(self, values):
        if not isinstance(values, list) or not (
            len(values) == 1 or all(len(row) == 1 for row in values)
        ):
            raise ValueError("Vector invalid type or shape")
        super().__init__(values)

    def dot(self, v):
        if not isinstance(v, Vector) or self.shape[1] != v.shape[0]:
            raise ValueError("Vector invalid type or shape")
        result = 0
        for i in range(self.shape[0]):
            result += self.data[i][0] * v.data[i][0]
        return result

    def __add__(self, other):
        result = super().__add__(other)
        return Vector(result.data)

    def __sub__(self, other):
        result = super().__sub__(other)
        return Vector(result.data)

    def __mul__(self, other):
        result = super().__mul__(other)
        if isinstance(result, Matrix):
            return Vector(result.data)
        return result

    def __truediv__(self, scalar: Number):
        result = super().__truediv__(scalar)
        return Vector(result.data)

    def __rtruediv__(self, scalar: Number):
        result = super().__rtruediv__(scalar)
        return Vector(result.data)

    def __str__(self):
        return f"{type(self)}: {str(self.data)}"

    def __repr__(self):
        return f"{type(self)}: {str(self.data)}"


if __name__ == "__main__":
    m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    print(m1.shape)
    # Output:
    print(f"Expected: (3, 2)")
    print(m1.T())
    # Output:
    print(f"Expected :Matrix([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])")
    print(m1.T().shape)
    # Output:
    print(f"Expected: (2, 3)")

    m1 = Matrix([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
    print(m1.shape)
    # Output:
    print(f"Expected: (2, 3)")
    print(m1.T())
    # Output:
    print(f"Expected: Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])")
    print(m1.T().shape)
    # Output:
    print(f"Expected: (3, 2)")

    m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
    m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    print(m1 * m2)
    # Output:
    print(f"Expected: Matrix([[28.0, 34.0], [56.0, 68.0]])")

    m1 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
    v1 = Vector([[1], [2], [3]])
    print(m1 * v1)
    # Output:
    print(f"Expected: Vector([[8], [16]])")

    v1 = Vector([[1], [2], [3]])
    v2 = Vector([[2], [4], [8]])
    print(v1 + v2)
    # Output:    # Or: Vector([[8], [16]

    print(f"Expected: Vector([[3], [6], [11]])")
