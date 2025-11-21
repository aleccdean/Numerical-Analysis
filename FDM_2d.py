#Build A matrix
import numpy as np

def toy_problem(x,y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def f(x,y):
    return 2*(np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)

def FDM(n, h):
    #Same as main but returns A, b and does not print
    U = np.zeros(n**2, dtype=object)
    x = 0
    for i in range(1,n+1):
        for j in range(1,n+1):
            U[x] = (j, i)
            x += 1
    index_map = {tuple(U[k]): k for k in range(U.shape[0])}
    A = np.zeros((U.shape[0], U.shape[0]))
    y = 0
    for i in range(1,n+1):
        for j in range(1,n+1):
            center_idx = index_map[(j, i)]
            A[y, center_idx] = 4

            if 1 <= j+1 <= n:
                A[y, index_map[(j+1, i)]] = -1
            if 1 <= j-1 <= 4:
                A[y, index_map[(j-1, i)]] = -1
            if 1 <= i+1 <= 4:
                A[y, index_map[(j, i+1)]] = -1
            if 1 <= i-1 <= 4:
                A[y, index_map[(j, i-1)]] = -1
            y += 1

    #Calculate b vec
    b = np.zeros(n**2)
    i = 0
    for node in index_map:
        b[i] = toy_problem(node[0],node[1])
        i +=1
    
    return A, b

def main(n,h):
    U = np.zeros(n**2, dtype=object)
    x = 0
    for i in range(1,n+1):
        for j in range(1,n+1):
            U[x] = (j, i)
            x += 1
    index_map = {tuple(U[k]): k for k in range(U.shape[0])}
    A = np.zeros((U.shape[0], U.shape[0]))
    y = 0
    for i in range(1,n+1):
        for j in range(1,n+1):
            center_idx = index_map[(j, i)]
            A[y, center_idx] = 4

            if 1 <= j+1 <= n:
                A[y, index_map[(j+1, i)]] = -1
            if 1 <= j-1 <= 4:
                A[y, index_map[(j-1, i)]] = -1
            if 1 <= i+1 <= n:
                A[y, index_map[(j, i+1)]] = -1
            if 1 <= i-1 <= 4:
                A[y, index_map[(j, i-1)]] = -1
            y += 1

    #print(A)

    #Calculate b vec
    b = np.zeros(n**2)
    exact = np.zeros(n**2)
    for k, (j,i) in enumerate(U):
        x = j * h
        y = i * h
        b[k] = f(x, y) * (h**2)
    i = 0
    for node in index_map:
        exact[i] = toy_problem(node[0],node[1])
        i +=1
   
    sol_vec = np.linalg.solve(A, b*(h**2))
    return sol_vec, exact

def error_chart():
    last = 0
    print(f"\n")
    print("  n    |   Error Value    |   Convergence Rate")
    print("----------------------------------------------")
    for i in range(7):
        n = 2 ** (i+1)
        h = 1/n
        sol_vec, exact_sol = main(n, h)
        error_vec = exact_sol - sol_vec
        error_val = np.max(np.abs(error_vec))
        if i > 0:
            convergence = np.log(last/error_val) / np.log(2.0)
        else:
            convergence = 0.0
        print(f"{n:4}   |  {error_val:13.8}   | {convergence:12.8}")
        last = error_val
    print(f"\n")
    

if __name__ == '__main__':
        #main(n, h)
        error_chart()