import numpy as np

class CustomPCA:
    def __init__(self, U, S, Vh):
        self.U = U
        self.S = S
        self.Vh = Vh
        self.V = Vh.conj().T  # Conjugate transpose of Vh
        self.explained_variance_ = (np.abs(self.S)**2) / (len(self.S) - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()*100
        self.cumulative_explained_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

    def transform(self, X):
        """
        Project the data X onto the principal components.
        The input X should be properly preprocessed (e.g., centered, scaled) if required.
        For complex data, ensure that X is compatible in type.
        """
        return np.dot(X, self.V)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the original data from its projection in the PCA space,
        using the complex conjugate transpose of V.
        """
        return np.dot(X_transformed, self.V.conj().T)

    @property
    def cum_exp_var(self):
        """
        Returns the cumulative explained variance ratio.
        """
        return self.cumulative_explained_variance_ratio_

# # Usage of the CustomPCA class
# # Assuming U_m, S_m, Vh_m are loaded from your file as shown previously

# # Example on how to use this class
# pca = CustomPCA(U_m, S_m, Vh_m)

# # Assuming you have some complex data matrix X to be transformed and then reconstructed
# X_transformed = pca.transform(X)
# print("Transformed Data:", X_transformed)

# X_reconstructed = pca.inverse_transform(X_transformed)
# print("Reconstructed Data:", X_reconstructed)
