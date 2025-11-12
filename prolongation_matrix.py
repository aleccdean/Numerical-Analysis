import numpy as np
import math
from scipy.sparse import csr_matrix


# Checking if the point is in the triangle
def pt_in_tri(pt, tri):
    x,y = pt
    ax, ay = tri[0]
    bx, by = tri[1]
    cx, cy = tri[2]

    side_1 = (x - bx) * (ay-by) -(ax-bx) *(y - by)
    side_2 = (x-cx) * (by - cy) - (bx - cx) * (y - cy)
    side_3 = (x - ax)* (cy - ay) - (cx - ax) * (y-ay)

    return (side_1 < 0) == (side_2 < 0) == (side_3 < 0)

# creating an array of points in the triangle
def in_tri(tri,node1, ele1, node2, ele2):
    pts = []
    n1, n2, n3 = ele1[tri]
    p1 = np.array([node1[n1, 0], node1[n1, 1]])
    p2 = np.array([node1[n2, 0], node1[n2, 1]])
    p3 = np.array([node1[n3, 0], node1[n3, 1]])
    for i in range(node2.shape[0]):
        if pt_in_tri(node2[i], [p1, p2, p3]):
            pts.append(i)
    return pts



def pr_matrix(node1, ele1, node2, ele2):
  ele1 -=1
  ele2 -=1

  N_node1 = node1.shape[0]
  N_node2 = node2.shape[0]

  N_ele1 = ele1.shape[0]
  N_ele2 = ele2.shape[0]

  #initializing a zero matrix
  PR = np.zeros((N_node2, N_node1))

  for k in range(N_ele1):
    #getting the nodes of the specfic triangle
    v1 = np.array(node1[ele1[k,0]])
    v2 = np.array(node1[ele1[k,1]])
    v3 = np.array(node1[ele1[k,2]])

    #creating a matrix p to create a matrix system to produce our basis functions
    p = np.array([[v1[0], v1[1], 1],
                 [v2[0], v2[1], 1],
                 [v3[0], v3[1], 1]])
   
    abc = np.zeros((3,3))

    # getting the coefficients for pur basis functions
    abc[0] = np.linalg.solve(p, [1,0,0])
    abc[1] = np.linalg.solve(p, [0,1,0])
    abc[2] = np.linalg.solve(p, [0,0,1])


    # initializing basis functions and putting them in an array
    phi_0 = lambda x,y: abc[0][0]*x + abc[0][1]*y + abc[0][2]
    phi_1 = lambda x,y: abc[1][0]*x + abc[1][1]*y + abc[1][2]
    phi_2 = lambda x,y: abc[2][0]*x + abc[2][1]*y + abc[2][2]

    phi_array = [phi_0, phi_1, phi_2]

    #Getting the points to evaluate
    pts = in_tri(k,node1, ele1, node2, ele2)

    assigned = np.zeros(N_node2, dtype=bool)

    for j in pts:
        #  avoiding double counting
        if assigned[j]:
            continue

        xj, yj = node2[j]
        for i in range(3):
            index = ele1[k, i]
            #evaluating the phi functions at the nodes in node2 file
            PR[j, index] = phi_array[i](xj, yj)
        assigned[j] = True

  return(PR)


if __name__ == '__main__':
  node1 = np.loadtxt('mesh_unitsquare/node1.txt')
  ele1 = np.loadtxt('mesh_unitsquare/ele1.txt').astype(int)
  node2 = np.loadtxt('mesh_unitsquare/node2.txt')
  ele2 = np.loadtxt('mesh_unitsquare/ele2.txt').astype(int)
  print(pr_matrix(node1, ele1, node2, ele2))