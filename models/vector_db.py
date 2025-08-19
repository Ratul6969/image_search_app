# models/vector_db.py
import os
from annoy import AnnoyIndex
import numpy as np
# Import paths from config
from config import ANNOY_INDEX_PATH, HANDLE_PATH, DIMENSION_PATH


class VectorDB:
    """Manages Annoy vector index and product handles."""
    
    def __init__(self):
        self.index = None
        self.handles = [] 
        self._loaded_dimension = None # To store the dimension once loaded/built
    
    def build_index(self, feature_vectors, handles, dimension):
        """
        Builds the Annoy index from a list of feature vectors and associated handles.
        Also saves the dimension.
        
        Args:
            feature_vectors (list of np.ndarray): List of 1D NumPy arrays (feature vectors).
            handles (list of str): List of corresponding product handles/IDs.
            dimension (int): The expected dimension of the feature vectors.
        """
        if not feature_vectors:
            raise ValueError("No feature vectors provided to build the index.")

        print(f"Building Annoy index with {len(feature_vectors)} items and dimension {dimension}...")
        self.index = AnnoyIndex(dimension, metric='angular') # Use angular for cosine similarity which is better for image features
        for i, vector in enumerate(feature_vectors):
            self.index.add_item(i, vector)
        
        self.index.build(n_trees=100) # Increased trees for better accuracy
        
        # Save the index, handles, and the dimension to disk
        self.index.save(ANNOY_INDEX_PATH) 
        np.save(HANDLE_PATH, handles)
        with open(DIMENSION_PATH, 'w') as f: # Save dimension to a text file for robust loading
            f.write(str(dimension))
        
        print(f"✅ Annoy index built and saved to {ANNOY_INDEX_PATH}")
        self.handles = handles 
        self._loaded_dimension = dimension # Store the dimension after building
    
    def load_index(self):
        """
        Loads a pre-built Annoy index, product handles, and dimension from disk.
        """
        # Check if all necessary index files exist
        if not os.path.exists(ANNOY_INDEX_PATH) or \
           not os.path.exists(HANDLE_PATH) or \
           not os.path.exists(DIMENSION_PATH):
            raise FileNotFoundError("Annoy index, handles, or dimension file not found. Please build the index first by running `python setup.py`.")
            
        print(f"Loading Annoy index from {ANNOY_INDEX_PATH}...")
        
        # Load the dimension first from the saved file
        with open(DIMENSION_PATH, 'r') as f:
            loaded_dim = int(f.read())
        self._loaded_dimension = loaded_dim # Store the loaded dimension

        # Initialize AnnoyIndex with the loaded dimension before loading the index file
        self.index = AnnoyIndex(self._loaded_dimension, metric='angular')
        self.index.load(ANNOY_INDEX_PATH) # Annoy will now load the index file expecting this dimension
        
        self.handles = np.load(HANDLE_PATH).tolist()
        print("✅ Annoy index and handles loaded successfully!")
    
    @property
    def loaded_dimension(self):
        """Returns the dimension of the currently loaded or built index."""
        return self._loaded_dimension

    def search(self, query_vector, k):
        """
        Finds the top-k most similar product handles for a given query vector.
        
        Args:
            query_vector (np.ndarray): The 1D feature vector of the query image.
            k (int): The number of nearest neighbors to return.
            
        Returns:
            list of str: A list of handles for the top-k matching products.
        """
        if self.index is None:
            raise RuntimeError("Annoy index not loaded. Call load_index() or build_index() first.")
        
        # Ensure query vector dimension matches the loaded index's dimension
        if query_vector.shape[0] != self.loaded_dimension:
            raise ValueError(f"Query vector has wrong length (expected {self.loaded_dimension}, got {query_vector.shape[0]}). This might indicate a mismatch between the feature extractor and the built index. Please re-run `python setup.py`.")

        # Perform the search for nearest neighbors
        nn_indices = self.index.get_nns_by_vector(query_vector, k)
        return [self.handles[i] for i in nn_indices]
