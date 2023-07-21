from matrix import Vector#, Matrix


#m1 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
#print(m1.shape)
## Output:
#print(f"Expected: (3, 2)")
#print(m1.T())
## Output:
#print(f"Expected :Matrix([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])")
#print(m1.T().shape)
## Output:
#print(f"Expected: (2, 3)")


#m1 = Matrix([[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])
#print(m1.shape)
## Output:
#print(f"Expected: (2, 3)")
#print(m1.T())
## Output:
#print(f"Expected: Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])")
#print(m1.T().shape)
## Output:
#print(f"Expected: (3, 2)")


#m1 = Matrix([[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 4.0, 6.0]])
#m2 = Matrix([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
#print(m1 * m2)
## Output:
#print(f"Expected: Matrix([[28.0, 34.0], [56.0, 68.0]])")


#m1 = Matrix([[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]])
#v1 = Vector([[1], [2], [3]])
#print(m1 * v1)
## Output:
#print(f"Expected: Matrix([[8], [16]])")
## Or: Vector([[8], [16]


v1 = Vector([[1], [2], [3]])
v2 = Vector([[2], [4], [8]])
print(v1 + v2)
# Output:
print(f"Expected: Vector([[3], [6], [11]])")
