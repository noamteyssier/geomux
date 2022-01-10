import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from umap import UMAP
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"

class Geomux:
    
    def __init__(self, min_umi=5, scalar=0, n_jobs=4):
        """
        param:
            min_umi: int: minimum number of UMIs to consider a cell barcode
        """
        self.min_umi = min_umi
        self.scalar = scalar
        self.n_jobs = n_jobs
        self.is_fit = False
        self.is_predict = False
        
        self.param_M = None
        self.param_n = None
        self.param_N = None

        self.num_tests = None
    
    def log_initial(self):
        """
        Logs the provided parameters
        """
        print(
            f"""
            Fitting Model with Params:
                min_umi: {self.min_umi}
                scalar: {self.scalar}
                n_jobs: {self.n_jobs}
            """,
            file=sys.stderr
        )
    
    def log_size(self, shape, prefix=""):
        """
        Logs the current size of the matrix
        """
        print(
            f"{prefix}: {shape}",
            file=sys.stderr
        )
    
    def log_hg(self, shape):
        """
        Logs the matrix size for the hypergeometric tests
        """
        print(
            f"Performing Hypergeometric Tests for {shape[0]} cells and {shape[1]} guides",
            file=sys.stderr
        )
    
    def log_params(self):
        """
        Logs the fit model parameters
        """
        print(
            f"""
            Population Size (M): {self.param_M}\n
            Number of Success States (n):
            {self.param_n}\n
            Number of Draws (N): 
            {self.param_N}
            """,
            file=sys.stderr
        )

    def pivot(self, frame):
        matrix = pd.pivot_table(
            frame,
            index="barcode",
            columns="guide",
            values="n_umi").fillna(0)
        self.log_size(matrix.shape, prefix="Initial Size")
        return matrix

    def filter(self, matrix):
        filt_mat = matrix[matrix.sum(axis=1) > self.min_umi]
        self.log_size(filt_mat.shape, prefix="Filtered Size")
        return filt_mat

    def parameterize_hypergeometric_distribution(self, matrix):
        """
        determines the parameters for the hypergeometric distribution
        """
        self.param_M = matrix.sum().astype(int)
        self.param_n = matrix.sum(axis=0).astype(int)
        self.param_N = matrix.sum(axis=1).astype(int)
        self.log_params()

    def hg_test(self, vec, idx):
        """
        performs a hypergeometric test for a guide 
        vector for a specific cell {idx}.
        """
        return hypergeom.sf(
            vec,
            self.param_M,
            self.param_n,
            self.param_N[idx])
    
    def hypergeometric_test(self, matrix):
        """
        performs a hypergeometric test for all barcode~guide pairs
        """
        self.log_hg(matrix.shape)
        with Pool(self.n_jobs) as p:
            pv_mat = np.vstack(
                p.starmap(
                    self.hg_test,
                    zip(
                        matrix,
                        np.arange(matrix.shape[0])
                    )
                )
            )
        
        pv_mat = np.clip(pv_mat, np.min(pv_mat[pv_mat != 0]), 1)
        return pv_mat 

    def adjusted_pvalues(self) -> np.ndarray:
        """
        calculate the adjusted p-values using a 
        bonferonni correction
        """
        return np.clip(self.pv_mat * self.num_tests, 0, 1)
    
    def log_odds(self, matrix):
        """
        calculates log odds as the ratio between the majority
        and second majority guide in a cell.

        In the case of negative-log pvalues this will calculate
        the log 2 odds ratio between the observed significance
        """
        lor = np.zeros(matrix.shape[0])
        for i in np.arange(matrix.shape[0]):
            min_idx, maj_idx = np.argsort(matrix[i])[-2:]
            min_val, maj_val = matrix[i][[min_idx, maj_idx]]
            lor[i] = np.log2((maj_val+self.scalar) / (min_val+self.scalar))
        return lor

    def fit(self, frame):
        """
        Preprocesses dataframe then performs hypergeometric testing
        on barcode~guide combinations. 
        """
        self.log_initial()
        
        # Matrix Preprocessing
        matrix = self.pivot(frame)
        matrix = self.filter(matrix)
        
        # barcode IDs and guide IDs
        self.cells = matrix.index.values
        self.guides = matrix.columns.values

        # assign number of tests
        self.num_tests = self.cells.size * self.guides.size
        
        # hypergeometric testing
        self.parameterize_hypergeometric_distribution(matrix.values)
        self.pv_mat = self.hypergeometric_test(matrix.values)
        self.adj_pv_mat = self.adjusted_pvalues()
        
        # calculate log odds
        self.lor = self.log_odds(-np.log(self.pv_mat))
        
        # matrix statistics
        self.n_umi = matrix.sum(axis=1)
        self.m_umi = matrix.mean(axis=1)
        self.v_umi = matrix.var(axis=1)
        self.max_umi = matrix.max(axis=1)
        
        self.is_fit = True

    def predict(self, min_lor = 0.5):
        """
        Assigns each cell barcode to its top guide if the LOR
        passes the provided threshold
        """
        if not self.is_fit:
            raise AttributeError("Model Must first be fit")

        self.mask = self.lor > min_lor
        self._assignments = pd.DataFrame({
            "barcode": self.cells[self.mask],
            "guide": self.guides[self.pv_mat[self.mask].argmin(axis=1)],
            "lor": self.lor[self.mask],
            "pvalue": self.pv_mat[self.mask].min(axis=1),
            "adj_pvalue": self.adj_pv_mat[self.mask].min(axis=1),
            "max_umi": self.max_umi[self.mask],
            "n_umi": self.n_umi[self.mask],
            "m_umi": self.m_umi[self.mask],
            "v_umi": self.v_umi[self.mask],
            "log_max_umi": np.log10(self.max_umi[self.mask]),
            "log_n_umi": np.log10(self.n_umi[self.mask]),
            "log_m_umi": np.log10(self.m_umi[self.mask])
        })
        
        self.is_predict = True
    
    def assignments(self):
        """
        returns the predicted assignments 
        """
        if not self.is_predict:
            raise AttributeError("Module must first be predicted")
        
        return self._assignments
    
    def plot_umap(self):
        """
        creates a plotly of the UMAP representation of the assigned barcodes
        """
        if not self.is_predict:
            raise AttributeError("Module must first be predicted")
            
        um = UMAP(random_state=42, metric="correlation")
        um_scatter = um.fit_transform(self.pv_mat[self.mask])
        self._assignments["umap_x"] = um_scatter[:, 0]
        self._assignments["umap_y"] = um_scatter[:, 1]
        
        fig = px.scatter(
            self._assignments,
            x="umap_x",
            y="umap_y",
            color="guide",
            size="lor",
            opacity=0.5)
        fig.update_layout(height=800, width=800)
        return fig
    
    def plot_correlation(self):
        """
        crates a plotly of the correlation between `n_umi` and `lor`
        """
        if not self.is_predict:
            raise AttributeError("Module must first be predicted")
        fig = px.scatter(
            self._assignments, 
            x="n_umi", 
            y="lor", 
            opacity=0.5,
            color="log_max_umi", 
            log_x=True)
        fig.update_layout(height=800, width=800)
        return fig
