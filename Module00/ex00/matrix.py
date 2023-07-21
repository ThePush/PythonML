from dataclasses import dataclass, field
from numbers import Number
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] \
        in %(funcName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Matrix:
    data: list = field(default_factory=list)
    shape: tuple((int, int)) = field(default_factory=tuple)

    def __init__(self, values):
        try:
            if isinstance(values, list):
                if not all(isinstance(x, list) for x in values) or not all(
                    isinstance(y, Number) for x in values for y in x
                ):
                    raise Exception("Matrix invalid list type")
                if len(values) > 1:
                    for row in values:
                        if len(row) != len(values[0]):
                            raise Exception("Matrix invalid list size")
                self.data = values
                logger.debug("Matrix list")
            elif isinstance(values, tuple):
                if (
                    not all(isinstance(x, int) for x in values)
                    or type(values[0]) != type(values[1])
                    or values[0] >= values[1]
                    or len(values) != 2
                ):
                    raise Exception("Matrix invalid tuple range")
                logger.debug("Matrix tuple")
                self.data = []
                for i in range(values[0]):
                    self.data.append([])
                    for _ in range(values[1]):
                        self.data[i].append(0.0)
            else:
                raise Exception("Matrix invalid type")
            self.shape = (len(self.data), len(self.data[0]))
            logger.debug("Matrix success")
            logger.debug(f"Matrix shape: {self.shape}")
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def __add__(self, other):
        try:
            if not isinstance(other, Matrix):
                raise Exception("Matrix invalid type")
            if self.shape != other.shape:
                raise Exception("Matrix invalid shape")
            result = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][j] = self.data[i][j] + other.data[i][j]
            return result
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        try:
            if not isinstance(other, Matrix):
                raise Exception("Matrix invalid type")
            if self.shape != other.shape:
                raise Exception("Matrix invalid shape")
            result = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][j] = self.data[i][j] - other.data[i][j]
            return result
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def __rsub__(self, other):
        try:
            if not isinstance(other, Matrix):
                raise Exception("Matrix invalid type")
            if self.shape != other.shape:
                raise Exception("Matrix invalid shape")
            result = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][j] = other.data[i][j] - self.data[i][j]
            return result
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def __truediv__(self, scalar: Number):
        try:
            if not isinstance(scalar, Number):
                raise Exception("Matrix invalid type")
            result = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][j] = self.data[i][j] / scalar
            return result
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def __rtruediv__(self, scalar: Number):
        try:
            if not isinstance(scalar, Number):
                raise Exception("Matrix invalid type")
            result = Matrix(self.shape)
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[i][j] = scalar / self.data[i][j]
            return result
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def __mul__(self, other):
        try:
            if isinstance(other, Number):
                result = Matrix(self.shape)
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        result.data[i][j] = self.data[i][j] * other
                return result
            elif isinstance(other, Vector):
                if self.shape[1] != other.shape[0]:
                    raise Exception("Matrix invalid shape")
                result = Vector((self.shape[0], 1))
                for i in range(self.shape[0]):
                    for j in range(self.shape[1]):
                        result.data[i][0] += self.data[i][j] * other.data[j][0]
                return result
            elif isinstance(other, Matrix):
                if self.shape[1] != other.shape[0]:
                    raise Exception("Matrix invalid shape")
                result = Matrix((self.shape[0], other.shape[1]))
                for i in range(self.shape[0]):
                    for j in range(other.shape[1]):
                        for k in range(self.shape[1]):
                            result.data[i][j] += self.data[i][k] * other.data[k][j]
                return result
            else:
                raise Exception("Matrix invalid type")
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def T(self):
        try:
            result = Matrix((self.shape[1], self.shape[0]))
            for i in range(self.shape[0]):
                for j in range(self.shape[1]):
                    result.data[j][i] = self.data[i][j]
            return result
        except Exception as e:
            logger.error(type(e).__name__ + ": " + str(e))
            return

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        return f"Matrix: {str(self.data)}"

    def __repr__(self):
        return f"Matrix: {str(self.data)}"


@dataclass
class Vector(Matrix):
    # create a Vector class that inherit from Matrix. At initialization, you  must check that a column or a row vector is passed as the data argument. If not, you must send an error message


if __name__ == "__main__":
    m = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
    m2 = Matrix((3, 5))
    print(m.data)
    print(m.shape)
    print(m2.data)
    print(m2.shape)
    sys.exit(0)
