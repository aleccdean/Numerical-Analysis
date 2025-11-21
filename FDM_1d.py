import numpy as np
import matplotlib.pyplot as plt

def exact_solution(x):
    return x**4 + x

def FDM_1D(rhs, c, d, domain, num_h):
    a, b = domain
    A = np.zeros((num_h, num_h))
    b_vec = np.zeros(num_h)
    h = (b-a)/(num_h+1) #0.25 in toy problem with num_h=3
    for i in range(num_h):
        x_i = a + h*(i+1) #0.25,0.5,0.75
        for j in range(num_h):
            if i==j:
                A[i][j] = -2
            elif np.abs(i-j) == 1:
                A[i][j] = 1
            else:
                A[i][j] = 0
        b_vec[i] += rhs(x_i,h)
        if i == 0:
            b_vec[i] += -c
        if i == num_h - 1:
            b_vec[i] += -d
    solution_vector = np.linalg.solve(A,b_vec)

    x = np.linspace(a, b, num_h + 2) #Domain values
    #Create full solution vector u, that includes boundary conditions
    u = np.zeros(num_h+2) 
    u[0] = c
    u[-1] = d
    u[1:-1] = solution_vector

    return x, u

def plot_exact():
  x = np.linspace(domain[0],domain[1], 501)
  exact = exact_solution(x)
  plt.figure()
  plt.plot(x, exact, linestyle='-', label='Exact')
  plt.title(f"FDM 1D Exact Solution")
  plt.xlabel("x")
  plt.ylabel("u(x)")
  plt.grid(True)
  plt.show()

def plot_solution(x,u, h):
  exact = exact_solution(x)
  plt.figure()
  plt.plot(x,u, marker='o', linestyle='-')
  plt.title(f"FDM 1D Solution: h={h}")
  plt.xlabel("x")
  plt.ylabel("u(x)")
  plt.grid(True)
  plt.show()

if __name__ == '__main__':
    rhs = lambda x,h: 12 * (x**2) * (h**2)
    #boundary: u(a) = c, u(b)=d
    c = 0
    d = 2
    domain = (0,1) #(a,b)
    plot_exact()
    for i in range(6):
        num_h = 2**(i+2)
        h = (domain[1]-domain[0])/num_h
        x, u = FDM_1D(rhs, c, d, domain, num_h)
        plot_solution(x, u, h)
    
