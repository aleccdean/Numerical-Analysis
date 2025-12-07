import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def exact(x, t):
    # Exact solution for error checking
    return (np.e ** (-(np.pi ** 2) * t)) * np.sin(np.pi * x)



def coefficient_matrix(num_h):
    A = np.zeros((num_h, num_h))


    for i in range(num_h):
        for j in range(num_h):
            if i==j:
                A[i][j] = -2
            elif np.abs(i-j) == 1:
                A[i][j] = 1
            else:
                A[i][j] = 0
    return A

def rhs(t, u, Dxx, boundary):
    u0, uN = boundary
    N = len(u)
    h = 1.0 / (N + 1)
    b = np.zeros(N)
    b[0] += u0 / h**2
    b[-1] += uN / h**2
    return Dxx @ u / h**2 + b

def main(a, b, c, d, num_h):
    x = np.linspace(a, b, num_h + 2) #Domain values
    Dxx = coefficient_matrix(num_h) 
    
    u0 = np.sin(np.pi * x[1:-1]) #Initial condition 
    time_span = (0, 0.1)

    sol = solve_ivp(rhs, time_span, u0, args=(Dxx, (c,d))) #t_eval=np.linspace(time_span[0],time_span[1],num_h+2)
    u = np.zeros((num_h+2, sol.y.shape[1]))
    u[0, :] = c
    u[-1, :] = d
    u[1:-1, :] = sol.y

    u_exact = exact(x, sol.t[-1])

    return x, u, u_exact

def plot_exact(a, b, t):
  x = np.linspace(a,b, 501)
  exact_sol = exact(x,t)
  plt.figure()
  plt.plot(x, exact_sol, linestyle='-', label='Exact',color='r')
  plt.title(f"Exact Solution")
  plt.xlabel("x")
  plt.ylabel("u(x)")
  plt.grid(True)
  plt.show()

def plot_solution(x,u, h):
  plt.figure()
  plt.plot(x,u, marker='o', linestyle='-')
  plt.title(f"MOL 1D Solution: h={h}")
  plt.xlabel("x")
  plt.ylabel("u(x)")
  plt.grid(True)
  plt.show()

def error_chart(a,b,c,d):
    last = 0
    print(f"\n")
    print("  interior nodes    |   Error Value    |   Convergence Rate")
    print("-----------------------------------------------------------")
    for i in range(7):
        num_h = 2 ** (i+2)
        x, sol_vec, exact_sol = main(a,b,c,d,num_h)
        plot_solution(x, sol_vec[:, -1], h=(b-a)/(num_h+1))
        error_vec = exact_sol - sol_vec[:, -1]
        error_val = np.max(np.abs(error_vec))
        if i > 0:
            convergence = np.log(last/error_val) / np.log(2.0)
        else:
            convergence = 0.0
        print(f"     {num_h:10}     |  {error_val:13.8}   | {convergence:12.8}")
        last = error_val
    print(f"\n")
    
if __name__ == '__main__':
    #boundary: u(a) = c, u(b)=d
    c = 0
    d = 0
    a, b = (0,1)
    error_chart(a, b, c, d)
    #plot_exact(a, b, t=0.1)