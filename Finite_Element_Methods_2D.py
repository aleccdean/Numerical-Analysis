#Finite element methods on two-dimensional domains
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import prolongation_matrix as prolongation_matrix


def exact(x,y):
    return math.cos(math.pi*x)*math.cos(math.pi*y) #example function

def F(x,y):
    return (2* math.pi**2+1)*exact(x,y) #right hand side function for our example exact solution

def rhs(x,y):
   return math.sin(math.pi*x)*math.sin(math.pi*y)

def grad_exact(x,y):
    return np.array([-math.pi*math.sin(math.pi*x)*math.cos(math.pi*y), -math.pi*math.cos(math.pi*x)*math.sin(math.pi*y)])

# Find basis functions 
def find_basis_functions(v1, v2, v3):
    P = np.array([[v1[0], v1[1], 1],
                    [v2[0], v2[1], 1],
                    [v3[0], v3[1], 1]])
    phi0 = np.linalg.solve(P, np.array([1, 0, 0]))
    phi1 = np.linalg.solve(P, np.array([0, 1, 0]))
    phi2 = np.linalg.solve(P, np.array([0, 0, 1]))
    return phi0, phi1, phi2

# Gaussian quadrature 
def guassian_quadrature(v1, v2, v3, f):
    #Lengths of sides -correct
    a = np.linalg.norm(v2 - v3)
    b = np.linalg.norm(v1 - v3)
    c = np.linalg.norm(v1 - v2)
    #Midpoints - Correct
    m1 = (v2 + v3) / 2
    m2 = (v1 + v3) / 2
    m3 = (v1 + v2) / 2
    #Centroid of the triangle - correct
    centroid = (v1 + v2 + v3) / 3
    # Area of the triangle(Herons formula) - correct
    s = (a + b + c) / 2
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))

    #Guassian quadrature formula
    integral = (area / 60) * (3 * (f(v1[0], v1[1]) + f(v2[0], v2[1]) + f(v3[0], v3[1])) + 8 * (f(m1[0], m1[1]) + f(m2[0], m2[1]) + f(m3[0], m3[1])) + 27 * f(centroid[0], centroid[1]))
    return integral



def FEM(node, ele): #Inputs: F(x,y) (right hand side function), domain(omega), node file, element file,
    #Output U_h(x,y) "approx solution"
    A_global = np.zeros((node.shape[0], node.shape[0]))
    b = np.zeros(node.shape[0])
    basis_functions = ele.shape[0] *[0]
    for k in range(ele.shape[0]):
      #Basis functions for each element
      v1 = node[ele[k,0]-1]
      v2 = node[ele[k,1]-1]
      v3 = node[ele[k,2]-1]
      phi = find_basis_functions(v1, v2, v3)
      basis_functions[k] = phi
      A_local = np.zeros((3, 3))
      rows = np.zeros(3 * len(ele[0]), dtype=float)
      cols = np.zeros(3 * len(ele[0]), dtype=float)

      #Computes local matrix for each triangle
      for i in range(3):
        #Find b vector
        b[ele[k][i]-1] = b[ele[k][i]-1] + guassian_quadrature(v1, v2, v3, lambda x, y: rhs(x, y) * (phi[i][0] * x + phi[i][1] * y + phi[i][2]))
        #Find error
        for j in range(3):
          rows[3*i+j] = ele[k][i] -1
          cols[3*i+j] = ele[k][j] -1
          f = lambda x, y: (np.dot((phi[i][:2]), (phi[j][:2]))) + (phi[i][0] * x + phi[i][1] * y + phi[i][2]) * (phi[j][0] * x + phi[j][1] * y + phi[j][2])
          A_local[i,j] = guassian_quadrature(v1, v2, v3, f)

      entries = A_local.flatten()
      A_global += csr_matrix((entries, (rows, cols)), shape=(node.shape[0], node.shape[0])).toarray()
    return A_global, b, basis_functions

#UNUSED FUNCTION
def compute_b():
    node = np.loadtxt("mesh_unitsquare/node3.txt", dtype=float)
    ele = np.loadtxt("mesh_unitsquare/ele3.txt", dtype=int) #ele = ele-1
    b = np.zeros(node.shape[0])
    for k in range(ele.shape[0]):
        #Basis functions for each element
        v1 = node[ele[k,0]-1]
        v2 = node[ele[k,1]-1]
        v3 = node[ele[k,2]-1]
        
        #Computes local matrix for each triangle
        for i in range(3):
            phi = find_basis_functions(v1, v2, v3)
            b[ele[k][i]-1] = b[ele[k][i]-1] + guassian_quadrature(v1, v2, v3, lambda x, y: F(x, y) * (phi[i][0] * x + phi[i][1] * y + phi[i][2]))

def compute_error(node, ele, u, basis_functions):
  L2_error = 0
  H1_error = 0
  H1_semi_error = 0
  for k in range(ele.shape[0]):
    v1 = node[ele[k,0]-1]
    v2 = node[ele[k,1]-1]
    v3 = node[ele[k,2]-1]
    v = lambda x, y: (exact(x,y)-(
        u[ele[k][0]-1]*(basis_functions[k][0][0]*x + basis_functions[k][0][1]*y + basis_functions[k][0][2]) +
        u[ele[k][1]-1]*(basis_functions[k][1][0]*x + basis_functions[k][1][1]*y + basis_functions[k][1][2]) +
        u[ele[k][2]-1]*(basis_functions[k][2][0]*x + basis_functions[k][2][1]*y + basis_functions[k][2][2])
        ))**2
    
    #Calculate gradient of v
    grad_u_h = lambda x, y: (grad_exact(x,y)-(
        u[ele[k][0]-1]*(basis_functions[k][0][:2]) +
        u[ele[k][1]-1]*(basis_functions[k][1][:2]) +
        u[ele[k][2]-1]*(basis_functions[k][2][:2])
        ))

    grad_v = lambda x, y: np.dot(grad_exact(x,y), grad_u_h(x,y)) #Maybe **2
    v_grad_v = lambda x, y: v(x,y) + grad_v(x,y) 
    L2_error += guassian_quadrature(v1, v2, v3, v)
    H1_semi_error += guassian_quadrature(v1, v2, v3, grad_v) 
    H1_error += guassian_quadrature(v1, v2, v3, v_grad_v)


  L2_error = math.sqrt(L2_error)
  H1_error = math.sqrt(H1_error)
  H1_semi_error = math.sqrt(H1_semi_error)
  return L2_error, H1_error, H1_semi_error

def plot_solution(x,y,c):
  ax = plt.figure().add_subplot(projection='3d')
  ax.plot_trisurf(x, y, c)
  ax.set_title("FEM Solution")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  plt.show()


def exact_error_chart():
  #ADD ORDER of convergence: log base 2 of previous error divided next error
  #log_2(previous/next)
  L2_error = np.zeros(6)
  H1_error = np.zeros(6)
  H1_semi_error = np.zeros(6)
  L2_convergence = np.zeros(6)
  H1_convergence = np.zeros(6)
  H1_semi_convergence = np.zeros(6)
  print("Mesh |   L2 Error | L2 Convergence |  H1 error |  H1 convergence |  H1_semi error | H1_semi convergence")
  print("-------------------------------------------------------------------------------------------------------")
  for i in range(1,7):
    node = np.loadtxt(f"mesh_unitsquare/node{i}.txt", dtype=float)
    ele = np.loadtxt(f"mesh_unitsquare/ele{i}.txt", dtype=int)
    A_global, b, basis_functions = FEM(node, ele)
    x = node[:,0]
    y = node[:,1]
    u = np.linalg.solve(A_global, b)
    L2_error[i-1], H1_error[i-1], H1_semi_error[i-1] = compute_error(node, ele, u, basis_functions)
    if i > 1:
      L2_convergence[i-1] = np.log2((L2_error[i-2]/L2_error[i-1]))
      H1_convergence[i-1] = np.log2((H1_error[i-2]/H1_error[i-1]))
      H1_semi_convergence[i-1] = np.log2((H1_semi_error[i-2]/H1_semi_error[i-1]))
    print(f"{i:5}| {L2_error[i-1]:9.5}  | {L2_convergence[i-1]:9.5}    |  {H1_error[i-1]:9.5}   | {H1_convergence[i-1]:9.5}      | {H1_semi_error[i-1]:9.5}      | {H1_semi_convergence[i-1]:9.5} ")

#UNUSED FUNCTION
def unknown_h1_error(node1, ele1, node2, ele2):
  #H1-error
  A_global1, b1, _ = FEM(node1, ele1)
  A_global2, b2, _ = FEM(node2, ele2)
  c1 = np.linalg.solve(A_global1, b1)
  c2 = np.linalg.solve(A_global2, b2)
  P = prolongation_matrix.pr_matrix(node1, ele1.copy(), node2, ele2.copy())
  x = P@c1-c2
  H1_error = math.sqrt(x.T@A_global2@x)
  return H1_error

def compute_unkown_error(node1, ele1, node2, ele2):
  u1, b1, basis_functions1 = FEM(node1, ele1)
  u2, b2, basis_functions2 = FEM(node2, ele2)
  c1 = np.linalg.solve(u1, b1)
  c2 = np.linalg.solve(u2, b2)
  P = prolongation_matrix.pr_matrix(node1, ele1.copy(), node2, ele2.copy())
  L2_error = 0
  H1_error = 0
  H1_semi_error = 0
  X=P@c1-c2
  for k in range(ele2.shape[0]):
    v1 = node2[ele2[k,0]-1]
    v2 = node2[ele2[k,1]-1]
    v3 = node2[ele2[k,2]-1]
    v = lambda x, y: (
       (X[ele2[k][0]-1]*(basis_functions2[k][0][0]*x + basis_functions2[k][0][1]*y + basis_functions2[k][0][2]) +
        X[ele2[k][1]-1]*(basis_functions2[k][1][0]*x + basis_functions2[k][1][1]*y + basis_functions2[k][1][2]) +
        X[ele2[k][2]-1]*(basis_functions2[k][2][0]*x + basis_functions2[k][2][1]*y + basis_functions2[k][2][2])) 
        )**2
    
    #Calculate gradient of v
    grad_u_h = lambda x, y: (
       (
        X[ele2[k][0]-1]*(basis_functions2[k][0][:2]) +
        X[ele2[k][1]-1]*(basis_functions2[k][1][:2]) +
        X[ele2[k][2]-1]*(basis_functions2[k][2][:2])
        ))

    grad_v = lambda x, y: np.dot(grad_u_h(x,y), grad_u_h(x,y)) #Maybe **2
    v_grad_v = lambda x, y: v(x,y) + grad_v(x,y) 
    L2_error += guassian_quadrature(v1, v2, v3, v)
    H1_semi_error += guassian_quadrature(v1, v2, v3, grad_v) 
    H1_error += guassian_quadrature(v1, v2, v3, v_grad_v)

  L2_error = math.sqrt(L2_error)
  H1_error = math.sqrt(H1_error)
  H1_semi_error = math.sqrt(H1_semi_error)
  return L2_error, H1_semi_error, H1_error


def unkown_error_chart():
  #ADD ORDER of convergence: log base 2 of previous error divided next error
  #log_2(previous/next)
  L2_error = np.zeros(6)
  H1_error = np.zeros(6)
  H1_semi_error = np.zeros(6)
  L2_convergence = np.zeros(6)
  H1_convergence = np.zeros(6)
  H1_semi_convergence = np.zeros(6)
  print("Mesh |   L2 Error | L2 Convergence |  H1 error |  H1 convergence |  H1_semi error | H1_semi convergence")
  print("-------------------------------------------------------------------------------------------------------")
  for i in range(2,7):
    node1 = np.loadtxt(f"mesh_unitsquare/node{i-1}.txt", dtype=float)
    ele1 = np.loadtxt(f"mesh_unitsquare/ele{i-1}.txt", dtype=int)
    node2 = np.loadtxt(f"mesh_unitsquare/node{i}.txt", dtype=float)
    ele2 = np.loadtxt(f"mesh_unitsquare/ele{i}.txt", dtype=int)
    L2_error[i-1], H1_semi_error[i-1], H1_error[i-1] = compute_unkown_error(node1,ele1,node2,ele2)
    if i > 2:
      L2_convergence[i-1] = np.log2((L2_error[i-2]/L2_error[i-1]))
      H1_convergence[i-1] = np.log2((H1_error[i-2]/H1_error[i-1]))
      H1_semi_convergence[i-1] = np.log2((H1_semi_error[i-2]/H1_semi_error[i-1]))
    print(f"{i:5}| {L2_error[i-1]:9.5}  | {L2_convergence[i-1]:9.5}    |  {H1_error[i-1]:9.5}   | {H1_convergence[i-1]:9.5}      | {H1_semi_error[i-1]:9.5}      | {H1_semi_convergence[i-1]:9.5} ")


if __name__ == "__main__":
  node1 = np.loadtxt(f"mesh_unitsquare/node1.txt", dtype=float)
  ele1 = np.loadtxt(f"mesh_unitsquare/ele1.txt", dtype=int)
  node2 = np.loadtxt(f"mesh_unitsquare/node2.txt", dtype=float)
  ele2 = np.loadtxt(f"mesh_unitsquare/ele2.txt", dtype=int)

  print(f"\n\n")
  unkown_error_chart()
  print(f"\n\n")
  #exact_error_chart()
  for i in range(1, 7):
    node = np.loadtxt(f"mesh_unitsquare/node{i}.txt", dtype=float)
    ele = np.loadtxt(f"mesh_unitsquare/ele{i}.txt", dtype=int)
    A_global, b, basis_functions = FEM(node, ele)
    x = node[:,0]
    y = node[:,1]
    u = np.linalg.solve(A_global, b)
    plot_solution(x,y,u)




"""
#Plot errors:
plt.plot(range(1,7), L2_error, marker='o', label='L2 Error')
plt.plot(range(1,7), H1_error, marker='o', label='H1 Error')
plt.plot(range(1,7), H1_semi_error, marker='o', label='H1 Semi Error')
#plt.yscale("log")
plt.xlabel("Mesh level")
plt.ylabel("Error")
plt.title("Error vs Mesh level")
plt.grid(True)
plt.show()
"""
  
"""
node = np.loadtxt("mesh_unitsquare/node6.txt", dtype=float)
ele = np.loadtxt("mesh_unitsquare/ele6.txt", dtype=int)
A_global, b, basis_functions = main(node, ele)
x = node[:,0]
y = node[:,1]
u = np.linalg.solve(A_global, b)
plot_solution(x,y,u)
error = compute_error(node, ele, u, basis_functions)
print(f"error: {error}")
print(A_global)
print(b)
"""

"""
Node1, ele1 output:
[[ 1.08333333 -0.45833333  0.         -0.45833333]
 [-0.45833333  1.16666667 -0.45833333  0.08333333]
 [ 0.         -0.45833333  1.08333333 -0.45833333]
 [-0.45833333  0.08333333 -0.45833333  1.16666667]]
"""

"""
Node2, ele2 output:
[[ 1.02083333  0.          0.          0.         -0.48958333  0.
   0.         -0.48958333  0.        ]
 [ 0.          1.04166667  0.          0.         -0.48958333 -0.48958333
   0.          0.          0.02083333]
 [ 0.          0.          1.02083333  0.          0.         -0.48958333
  -0.48958333  0.          0.        ]
 [ 0.          0.          0.          1.04166667  0.          0.
  -0.48958333 -0.48958333  0.02083333]
 [-0.48958333 -0.48958333  0.          0.          2.0625      0.
   0.          0.02083333 -0.97916667]
 [ 0.         -0.48958333 -0.48958333  0.          0.          2.0625
   0.02083333  0.         -0.97916667]
 [ 0.          0.         -0.48958333 -0.48958333  0.          0.02083333
   2.0625      0.         -0.97916667]
 [-0.48958333  0.          0.         -0.48958333  0.02083333  0.
   0.          2.0625     -0.97916667]
 [ 0.          0.02083333  0.          0.02083333 -0.97916667 -0.97916667
  -0.97916667 -0.97916667  4.125     ]]

"""