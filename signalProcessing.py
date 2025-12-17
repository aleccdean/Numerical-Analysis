import numpy as np
import matplotlib.pyplot as plt


def construct_signal(t):
    clean_signal = np.sin(10 * np.pi * t)
    noise = 0.1 * np.random.randn(*t.shape)
    noisy_signal = clean_signal + noise
    return noisy_signal, clean_signal

def denoise(A, row_len=50, r=None):
    #Map matrix to 2D
    n = len(A)
    num_rows = n // row_len
    reshaped_matrix = A[:num_rows * row_len].reshape((num_rows, row_len))
    
    r = scree_plot(reshaped_matrix) #Find best rank approximation
    
    #Reconstruct rank r approximation
    U, S, V = np.linalg.svd(reshaped_matrix, full_matrices=False)
    S_denoised = np.zeros_like(S)
    S_denoised[:r] = S[:r]
    denoised_matrix = U @ np.diag(S_denoised) @ V
    return denoised_matrix.flatten()

def scree_plot(A):
    u, s, v = np.linalg.svd(A)
    S = (s ** 2) / np.sum(s ** 2)
    i = np.arange(1, len(s) + 1)
    deriv2 = np.diff(S, n=2)
    elbow = np.argmax(deriv2) + 2 #Absolute maximum point of 2nd deriv
    
    #plotting
    plt.plot(i, S, 'o-', linewidth=2)
    plt.axvline(x=elbow, color='black', linestyle='--', label=f" Threshold={elbow}")
    plt.title("Scree plot")
    plt.ylabel("S[i][i]")
    plt.xlabel("i")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return elbow

    

if __name__ == '__main__':
    t = np.linspace(0, 1, 2000)
    A_noisy, A_clean = construct_signal(t)
    A_denoised = denoise(A_noisy)
    plt.plot(t, A_noisy, label='Noisy')
    plt.title("Noisy signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    plt.plot(t, A_denoised, label='Denoised')
    plt.title("Denoised signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    plt.plot(t, A_clean, label='clean')
    plt.title("Original signal")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()
    
