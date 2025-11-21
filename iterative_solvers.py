import numpy as np
from Finite_Element_Methods_2D import FEM
from FDM_2d import FDM

def iterative_solver(alpha, A, f,iterations):
    #Jacobi but goes for a certain # of iterations and spectral radius
    print(f"\nalpha value: {alpha}")
    D = np.diag(A)
    B = alpha * np.linalg.inv(np.diag(D))
    u_k = np.zeros(A.shape[0])
    for i in range(iterations):
        r = f - A @ u_k
        e = B @ r
        u_k = u_k + e
        print(f"{i}: {np.linalg.vector_norm(f - (A @ u_k))}") 
    I = np.identity(A.shape[0])
    e_vals, _ = np.linalg.eig((I-B@A))
    print(f"spectral radius: {np.max(np.abs(e_vals))}\n") #Spectral radius
    return u_k

def jacobi(alpha, A, b,tolerance):
    D = np.diag(A)
    B = alpha * np.linalg.inv(np.diag(D))
    u_k = np.zeros(A.shape[0])
    convergence = float('inf')
    iterations = 0
    while(convergence>tolerance):
        r = b - A @ u_k
        e = B @ r
        u_k = u_k + e
        convergence = np.linalg.vector_norm(r)
        iterations += 1
    return u_k, iterations

def guassian_siedel(A, b, tolerance):
    D = np.triu(A)
    #print(D)
    B = np.linalg.inv(D)
    u_k = np.zeros(A.shape[0])
    I = np.identity(A.shape[0])
    convergence = float('inf')
    iterations = 0
    while(convergence>tolerance):
        r = b - A @ u_k
        e = B @ r
        u_k = u_k + e
        convergence = np.linalg.vector_norm(r)
        #print(convergence)
        iterations += 1
        #e_vals, _ = np.linalg.eig((I-B@A))
        #print(np.max(np.abs(e_vals)))
    
    return u_k, iterations

def conjugate_gradient(A, b, tolerance):
    size = len(b)
    x_n_1 = np.zeros(size, dtype=float)
    r_n_1 = b
    p_n_1 = r_n_1
    convergence = float('inf')
    iterations = 0
    while(convergence > tolerance):
        alpha_n = (r_n_1.T @ r_n_1) / (p_n_1.T @ A @ p_n_1) # step length
        x_n = x_n_1 + (alpha_n * p_n_1)                     # approximate solution
        r_n = r_n_1 - (alpha_n * A @ p_n_1)                 # residual
        convergence = np.linalg.vector_norm(r_n)
        beta_n = (r_n.T @ r_n) / (r_n_1.T @ r_n_1)          # imporvement this step
        p_n = r_n + (beta_n * p_n_1)                        # search direction
        x_n_1, r_n_1, p_n_1 = x_n, r_n, p_n
        iterations += 1
    return x_n, iterations

def error_chart():
    alpha = 1
    tolerance = 10**(-12) 
    print(f"\nChart representing number of iterations each method took to converge:")
    print("mesh level | Jacobi Convergence | Gauss-siedel convergence | Conjugate Gradient Convergence")
    print("-------------------------------------------------------------------------------------------")
    for i in range(1, 7):
        node = np.loadtxt(f"mesh_unitsquare/node{i}.txt", dtype=float)
        ele = np.loadtxt(f"mesh_unitsquare/ele{i}.txt", dtype=int)
        A_global, b, _ = FEM(node, ele)
        _, gs_iterations = guassian_siedel(A_global, b, tolerance)
        _, cg_iterations = conjugate_gradient(A_global, b, tolerance)
        _, jacobi_iterations = jacobi(alpha, A_global, b, tolerance)
        print(f"{i:7}    | {jacobi_iterations:13}      |  {gs_iterations:13}            |  {cg_iterations:13}")
    print(f"\n")

def main():
    #JACOBI solution part 4 of report2
    A, f = FDM(4, 1/4) 
    alpha = 1
    iterations = 20
    jacobi_sol = iterative_solver(1, A, f, iterations)

    error_chart()


if __name__ == '__main__':
    main()
