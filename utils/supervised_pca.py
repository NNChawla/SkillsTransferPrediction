from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np

class SupervisedPCA(BaseEstimator, TransformerMixin):
    """Supervised Principal Component Analysis.
    
    This implementation first selects features based on their correlation with
    the target variable, then applies PCA to the selected features.
    
    Parameters
    ----------
    n_components : int, optional (default=None)
        Number of components to keep. If None, keep all components.
    threshold : float, optional (default=None)
        Correlation threshold for feature selection. If None, use n_components
        to select top correlated features.
    
    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space.
    selected_features_ : array
        Indices of selected features.
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each component.
    scaler_ : StandardScaler
        Fitted StandardScaler instance.
    imputer_ : SimpleImputer
        Fitted SimpleImputer instance.
    """
    def __init__(self, n_components=None, threshold=None):
        if n_components is not None and n_components < 1:
            raise ValueError("n_components must be >= 1")
        self.n_components = n_components
        self.threshold = threshold
        self.components_ = None
        self.selected_features_ = None
        self.scaler_ = None
        self.imputer_ = None
        self.explained_variance_ratio_ = None
        
    def fit(self, X, y):
        print("\nSupervisedPCA Fit Process:")
        print(f"Input shape: {X.shape}")
        print(f"Number of components requested: {self.n_components}")
        
        if X.shape[1] < 1:
            raise ValueError("No features provided")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of samples")
        
        # Convert X to numeric type
        X = X.astype(float)
        
        # For classification, convert labels to numeric
        label_encoder = LabelEncoder()
        y_numeric = label_encoder.fit_transform(y)
        print(f"Unique labels in y: {np.unique(y)}")
        print(f"Unique values in y_numeric: {np.unique(y_numeric)}")
        
        # Calculate correlation scores
        correlations = []
        for i in range(X.shape[1]):
            mask = ~np.isnan(X[:, i])
            if np.sum(mask) > 1:
                try:
                    corr = abs(np.corrcoef(X[mask, i], y_numeric[mask])[0, 1])
                    if np.isnan(corr):
                        corr = 0
                except Exception as e:
                    print(f"Error calculating correlation for feature {i}: {str(e)}")
                    corr = 0
            else:
                corr = 0
            correlations.append(corr)
        
        # Print correlation summary
        print("\nFeature Correlations:")
        print(f"Max correlation: {max(correlations):.3f}")
        print(f"Min correlation: {min(correlations):.3f}")
        print(f"Mean correlation: {np.mean(correlations):.3f}")
        
        # Select features based on correlation
        if self.threshold is not None:
            self.selected_features_ = np.where(np.array(correlations) >= self.threshold)[0]
        else:
            # Use number of components to select top features
            n_select = self.n_components if self.n_components else X.shape[1]
            top_indices = np.argsort(correlations)[::-1][:n_select]
            self.selected_features_ = top_indices
        
        print(f"\nSelected {len(self.selected_features_)} features")
        print(f"Selected feature indices: {self.selected_features_}")
        
        # Transform data using selected features
        X_selected = X[:, self.selected_features_]
        
        # Handle missing values for PCA
        imputer = SimpleImputer(strategy='mean')
        X_selected = imputer.fit_transform(X_selected)
        
        # Standardize before PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        # Perform PCA
        n_components = min(self.n_components if self.n_components else X_selected.shape[1],
                         X_selected.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)
        
        self.components_ = pca.components_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.scaler_ = scaler
        self.imputer_ = imputer
        
        return self
        
    def transform(self, X):
        print("\nSupervisedPCA Transform:")
        print(f"Input shape: {X.shape}")
        
        X_selected = X[:, self.selected_features_]
        print(f"Shape after feature selection: {X_selected.shape}")
        
        return X_selected

    def get_feature_names_out(self, feature_names=None):
        """Get output feature names for transformation."""
        if not hasattr(self, 'components_'):
            raise ValueError("SupervisedPCA has not been fitted yet.")
        
        if feature_names is not None:
            selected_names = np.array(feature_names)[self.selected_features_]
        else:
            selected_names = [f"feature{i}" for i in self.selected_features_]
        
        return [f"spca{i}" for i in range(self.components_.shape[0])]