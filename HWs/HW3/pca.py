import numpy as np

data = np.array([[1, 1], [1, 2], [2, 1], [-1, -1], [-1, -2], [-2, -1]])

centered_data = data - np.mean(data, axis=0)                        # Center the data
covariance_matrix = np.cov(centered_data.T)                         # Compute the Covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)        # Compute Eigenvalues and Eigenvectors
principal_component = eigenvectors[:, np.argmax(eigenvalues)]       # Compute PC variables
transformed_data = np.dot(centered_data, principal_component)       # Project the data
one_dimensional_array = transformed_data.flatten()                  # Just in case the output matrix wasn't 1D

print(one_dimensional_array)
