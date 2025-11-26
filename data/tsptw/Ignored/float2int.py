import sys
import os

def fracLen(s):
    return len(s.split('.')[-1].rstrip('0')) if '.' in s else 0

def maxFracLen(l):
    return max(map(fracLen, l), default=0)

# Read all numbers
input_path = sys.argv[1]
with open(input_path) as f:
    numbers = [x for x in f.read().split()]

# Parse n, matrix, and pairs
n = int(numbers[0])
matrix_start = 1
matrix_end = matrix_start + n * n
matrix = numbers[matrix_start:matrix_end]
pairs_start = matrix_end
pairs_end = pairs_start + 2 * n
pairs = numbers[pairs_start:pairs_end]

# print(f"n = {n}")
# print(f"d = ({len(matrix)}) {matrix}")
# print(f"tw = ({len(pairs)}) {pairs}")

# Find scale
scale = 10 ** maxFracLen(matrix)

# 4) Scale values
matrix = [int(float(x) * scale) for x in matrix]
pairs = [int(float(x) * scale) for x in pairs]

# 5) Write result
output_path = os.path.splitext(input_path)[0] + ".txt"
with open(output_path, "w") as f:
    f.write(f"{n}\n")
    for i in range(n):
        for j in range(n):
            f.write("" if j == 0 else " ")
            f.write(str(matrix[i*n+j]))
        f.write("\n")
    for i in range(0,n):
        f.write(f"{pairs[i*2]} {pairs[i*2+1]}\n")
    f.write(f"# scale factor = {scale}\n")

print(f"Scaling {input_path} -> {output_path} with scale factor {scale}")
