"""
Noisy data matrix 

@author: alecdean
"""
import numpy as np
import matplotlib.pyplot as plt


def filter(A, r=15):
    u, s, v = np.linalg.svd(A_noisy)
    S = np.zeros((40, 100))
    for i in range(len(s)):
        if i < r:
            S[i][i] = s[i]
    return u @ S @ v

def construct_dataset():
    # Step 1: Define dimensions
    T, D = 40, 100  # time steps < spatial sites
    
    # Step 2: Define clean singular vectors and values
    U = np.zeros((T, T))
    V = np.zeros((D, T))
    for t in range(T):
        for k in range(T):
            U[t, k] = np.cos((2 * np.pi / T) * (t - 1) * (k -1) - (np.pi / 4)) * np.sqrt(2 / T)
    
    for d in range(D):
        for k in range(T):
            V[d, k] = np.sin(np.pi * (d + 1) * (k + 1) / (D + 1)) * np.sqrt(2 / (D + 1))
    
    # Exponential decay of singular values
    s = np.array([10**(-4 * (k-1) / (T - 1)) for k in range(T)])
    S = np.diag(s)
    
    # Step 3: Form clean matrix
    A = U @ S @ V.T  # A is T x D
    return A


def add_noise(A):
    # Step 4: Add Gaussian noise
    epsilon = 10e-4
    noise = np.random.normal(0, epsilon, size=A.shape)
    A_noisy = A + noise
    return A_noisy


if __name__ == '__main__':
    A = construct_dataset()
    A_noisy = add_noise(A)
    A_denoised = filter(A_noisy)
    # Optional: visualize
    plt.imshow(A, aspect='auto', cmap='viridis', norm='log')
    plt.title("Clean Data Matrix A")
    plt.colorbar(label="Value")
    plt.xlabel("Data Site")
    plt.ylabel("Time Step")
    plt.show()
    
    
    plt.imshow(A_noisy,aspect='auto', cmap='viridis', norm='log')
    plt.title("Noisy Data Matrix $\~A$")
    plt.colorbar(label="Value")
    plt.xlabel("Data Site")
    plt.ylabel("Time Step")
    plt.show()
    
    plt.imshow(A_denoised,aspect='auto', cmap='viridis', norm='log')
    plt.title("Denoised Data Matrix $\~A$")
    plt.colorbar(label="Value")
    plt.xlabel("Data Site")
    plt.ylabel("Time Step")
    plt.show()
    
    
    u, s, v = np.linalg.svd(A)
    u_noisy, s_noisy, v_noisy = np.linalg.svd(A_noisy)
    plt.plot(s)
    plt.plot(s_noisy)
    
    
