import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

def makeHomogene(vertices):
    newVertices = []
    for v in vertices:
        newVertices.append([v[0], v[1], 1])
    return newVertices

def makeMatrixEquations3f(A,B,C):
    return np.array([
                     [A[0], B[0], C[0]],
                     [A[1], B[1], C[1]],
                     [A[2], B[2], C[2]]])

def matrix_P_for_NaiveAlgorithm(A,B,C,D):
    # Making equations
    equations = makeMatrixEquations3f(A,B,C)
    dp = np.array(D)

    # Solution of equation
    solution = np.linalg.solve(equations, dp)

    p = np.transpose(np.array([
        [solution[0]*A[0], solution[0]*A[1], solution[0]*A[2]],
        [solution[1]*B[0], solution[1]*B[1], solution[1]*B[2]],
        [solution[2]*C[0], solution[2]*C[1], solution[2]*C[2]]]))

    return p

def naive_algorithm(A,B,C,D,Ap,Bp,Cp,Dp):
    # Make homogeneous vertices
    vHomo = makeHomogene([A,B,C,D,Ap,Bp,Cp,Dp])

    p1 = matrix_P_for_NaiveAlgorithm(vHomo[0], vHomo[1], vHomo[2], vHomo[3])
    p2 = matrix_P_for_NaiveAlgorithm(vHomo[4], vHomo[5], vHomo[6], vHomo[7])

    p = np.dot(p2, np.linalg.inv(p1))
    return p

def makeMatrix2x9f(v):
    A = np.array([
                  [0, 0, 0, -v[1][2]*v[0][0], -v[1][2]*v[0][1], -v[1][2]*v[0][2], v[1][1]*v[0][0], v[1][1]*v[0][1], v[1][1]*v[0][2] ],
                  [v[1][2]*v[0][0], v[1][2]*v[0][1], v[1][2]*v[0][2], 0, 0, 0, -v[1][0]*v[0][0], -v[1][0]*v[0][1], -v[1][0]*v[0][2]]
                  ])
    return A

def dlt_algorithm(vertices):

    n = len(vertices)

    # Make homogeneous coordinates
    newVertices = []
    for v in vertices:
        newVertices.append(([v[0][0], v[0][1], 1] , [v[1][0], v[1][1], 1]))

    # Making 2*n x 9 matrix
    A = makeMatrix2x9f(newVertices[0])
    for i in range(1, n):
        tmp = makeMatrix2x9f(newVertices[i])
        A = np.concatenate((A, tmp), axis = 0)

    # Apply SVD decomposition on matrix A
    u, d, v = np.linalg.svd(A, full_matrices=False)

    # Take the last row of matrix v
    v = v[len(v)-1, :]
    p = np.array([
        [v[0], v[1], v[2]],
        [v[3], v[4], v[5]],
        [v[6], v[7], v[8]]
    ])
    return p

def euclid_distance(A, B):
    return np.sqrt( ((A[0]-B[0])**2) + ((A[1]-B[1])**2) )

def normalizationMatrix(vertices):
    n = len(vertices)

    # Counting center of "gravity" t of a system
    sum_x = 0
    sum_y = 0
    for v in vertices:
        sum_x += v[0]
        sum_y += v[1]

    t = np.array([sum_x/n, sum_y/n])

    # Translation matrix G
    G = np.array([
        [1, 0, -t[0]],
        [0, 1, -t[1]],
        [0, 0, 1]
    ])

    # Calculating distances from vertices to t
    di = []
    for v in vertices:
        di.append(euclid_distance(t, v))

    # Getting average distance
    sum = reduce(lambda x, y: x+y, di)
    avg = sum/n

    # Matrix of homotopy
    S = np.array([
        [np.sqrt(2)/avg, 0, 0],
        [0, np.sqrt(2)/avg, 0],
        [0, 0, 1]
      ])

    T = np.dot(S, G)
    return T

def applyTransformationOnVertices(T, vertices):
    newInputVertices = []
    for v in vertices:
        newInputVertices.append(np.dot(T, np.transpose(v)))
    return newInputVertices

def normalized_dlt_algorithm(inputVertices, outputVertices):

    # Getting T matrix of normalization
    T = normalizationMatrix(inputVertices)
    Tp = normalizationMatrix(outputVertices)

    # Making 3D coordinates out of 2D given
    inputVertices3D = makeHomogene(inputVertices)
    outputVertices3D = makeHomogene(outputVertices)

    # Applying transformation T on vertices
    newInputVertices = applyTransformationOnVertices(T, inputVertices3D)
    newOutputVertices = applyTransformationOnVertices(Tp, outputVertices3D)

    # Preparing vertices for entry for regular DLT as a list of tupples
    n = len(inputVertices)
    vertices = []
    for i in range(0, n):
        vertices.append((newInputVertices[i], newOutputVertices[i]))

    # Matrix transformation p' for regular dlt
    pp = dlt_algorithm(vertices)

    # Matrix p as a result of our algorithm
    p = np.dot(np.dot(np.linalg.inv(Tp),pp), T)
    return p

def printVertices(A,B,C,D,Ap,Bp,Cp,Dp):
   print("A = [" + str(A[0]) + ", " + str(A[1]) + "]")
   print("B = [" + str(B[0]) + ", " + str(B[1]) + "]")
   print("C = [" + str(C[0]) + ", " + str(C[1]) + "]")
   print("D = [" + str(D[0]) + ", " + str(D[1]) + "]")
   print("A' = [" + str(Ap[0]) + ", " + str(Ap[1]) + "]")
   print("B' = [" + str(Bp[0]) + ", " + str(Bp[1]) + "]")
   print("C' = [" + str(Cp[0]) + ", " + str(Cp[1]) + "]")
   print("D' = [" + str(Dp[0]) + ", " + str(Dp[1]) + "]")

def printVertices(A,B,C,D,E,F,Ap,Bp,Cp,Dp,Ep,Fp):
    print("A = [" + str(A[0]) + ", " + str(A[1]) + "]")
    print("B = [" + str(B[0]) + ", " + str(B[1]) + "]")
    print("C = [" + str(C[0]) + ", " + str(C[1]) + "]")
    print("D = [" + str(D[0]) + ", " + str(D[1]) + "]")
    print("E = [" + str(E[0]) + ", " + str(E[1]) + "]")
    print("F = [" + str(F[0]) + ", " + str(F[1]) + "]")

    print("A' = [" + str(Ap[0]) + ", " + str(Ap[1]) + "]")
    print("B' = [" + str(Bp[0]) + ", " + str(Bp[1]) + "]")
    print("C' = [" + str(Cp[0]) + ", " + str(Cp[1]) + "]")
    print("D' = [" + str(Dp[0]) + ", " + str(Dp[1]) + "]")
    print("E' = [" + str(Ep[0]) + ", " + str(Ep[1]) + "]")
    print("F' = [" + str(Fp[0]) + ", " + str(Fp[1]) + "]")

def main():
    A = [-5,0]
    B = [-6,0.5]
    C = [-5, 3]
    D = [-4, 2]
    E = [-2,1.5]
    F = [-4,0]

    Ap = [3,0]
    Bp = [1, 1.5]
    Cp = [2.7, 2.5]
    Dp = [6,2]
    Ep = [6.5,0.5]
    Fp = [4,0]

    print('\n')
    p = naive_algorithm(A,B,C,D,Ap,Bp,Cp,Dp)
    print("Naive algorithm: Transformation matrix P is ")
    print(p)
    print('\n')

    vertices = [(A,Ap), (B,Bp), (C,Cp), (D,Dp), (E,Ep), (F,Fp)]
    p = dlt_algorithm(vertices)
    print("DLT algorithm: Transformation matrix P is ")
    print(p)
    print('\n')

    inputVertices = [A,B,C,D,E,F]
    outputVertices = [Ap,Bp,Cp,Dp,Ep,Fp]
    p = normalized_dlt_algorithm(inputVertices, outputVertices)
    print("Normalized DLT algorithm: Transformation matrix P is ")
    print(p)
    print('\n')

if __name__ == "__main__":
    main()
