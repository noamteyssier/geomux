from multiprocessing import Pool
from typing import List, Union, Optional
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.stats import hypergeom
from scipy.special import logit
from adjustpy import adjust


class Geomux:
    def __init__(
        self,
        matrix: Union[np.ndarray, pd.DataFrame],
        cell_names: Optional[Union[List[str], np.ndarray, ArrayLike]] = None,
        guide_names: Optional[Union[List[str], np.ndarray, ArrayLike]] = None,
        min_umi: int = 5,
        min_cells: int = 100,
        n_jobs: int = 4,
        verbose: bool = False,
        method: str = "bh",
    ):
        """
        Parameters
        ----------
        matrix : np.ndarray
            matrix of cell x guide counts
        min_umi : int
            minimum number of UMIs to consider a cell barcode
        min_cells : int
            minimum number of cells to consider a guide
        n_jobs : int
            number of jobs to use for multiprocessing
        method: str
            pvalue adjustment procedure to use.
        """

        # Load the matrix
        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.values
        self.matrix = matrix

        # Load the cell and guide names
        if cell_names is None:
            cell_names = np.arange(matrix.shape[0])
        else:
            assert len(cell_names) == matrix.shape[0]
            cell_names = np.array(cell_names)

        if guide_names is None:
            guide_names = np.arange(matrix.shape[1])
        else:
            assert len(guide_names) == matrix.shape[1]
            guide_names = np.array(guide_names)

        self.cell_names = cell_names
        self.guide_names = guide_names

        # Set the parameters
        self.min_umi = min_umi
        self.min_cells = min_cells
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.method = method
        self._n_total = matrix.shape[0]
        self._m_total = matrix.shape[1]

        self._set_procedure()
        self._filter_matrix()
        self._fit_parameters()

        self._n_cells = self.matrix.shape[0]
        self._n_guides = self.matrix.shape[1]
        self._n_tests = self._n_cells * self._n_guides

        self.is_fit = False
        self.labels = []

    def _set_procedure(self):
        allowed_procedures = ["bonferroni", "bh", "by"]
        if self.method not in allowed_procedures:
            raise ValueError(
                f"Provided method {self.method} not recognized. Choose from {', '.join(allowed_procedures)}"
            )

    def _filter_matrix(self):
        """
        Filters the matrix to only include cells with at least
        `min_umi` UMIs
        """
        cell_sums = self.matrix.sum(axis=1)
        guide_sums = self.matrix.sum(axis=0)
        self.passing_cells = cell_sums >= self.min_umi
        self.passing_guides = guide_sums >= self.min_cells
        self.matrix = self.matrix[self.passing_cells][:, self.passing_guides]
        if self.verbose:
            print("---------------------------------------")
            print(
                "INFO: Average number of UMIs per cell: {:.2f}".format(cell_sums.mean())
            )
            print("INFO: Variance of UMIs per cell: {:.2f}".format(cell_sums.var()))
            print(
                "INFO: Cell UMI counts range from {} to {}".format(
                    cell_sums.min(), cell_sums.max()
                )
            )
            print("---------------------------------------")

            print(
                "INFO: Average number of cells per guide: {:.2f}".format(
                    guide_sums.mean()
                )
            )
            print("INFO: Variance of cells per guide: {:.2f}".format(guide_sums.var()))
            print(
                "INFO: Guide cell counts range from {} to {}".format(
                    guide_sums.min(), guide_sums.max()
                )
            )
            print("---------------------------------------")
            old_cell_size = self._n_total
            new_cell_size = self.matrix.shape[0]
            print(
                "LOG: Removed {} ({:.2f}%) cells with < {} UMIs".format(
                    old_cell_size - new_cell_size,
                    100 * (old_cell_size - new_cell_size) / old_cell_size,
                    self.min_umi,
                )
            )

            old_guide_size = self._m_total
            new_guide_size = self.matrix.shape[1]
            print(
                "LOG: Removed {} ({:.2f}%) guides with < {} Cells".format(
                    old_guide_size - new_guide_size,
                    100 * (old_guide_size - new_guide_size) / old_guide_size,
                    self.min_cells,
                )
            )
        if self.matrix.shape[0] == 0:
            raise ValueError(
                "No cells passed the UMI threshold. Try lowering the min_umi parameter"
            )
        if self.matrix.shape[1] == 0:
            raise ValueError(
                "No guides passed the cell threshold. Try lowering the min_cells parameter"
            )

    def _fit_parameters(self):
        """
        Fits the hypergeometric distribution parameters
        """
        self.population = self.matrix.sum()
        self.successes = self.matrix.sum(axis=0)
        self.draws = self.matrix.sum(axis=1)

    def _hypergeometric_test(self, x: np.ndarray, idx: int):
        """
        Calculates the survival function of a hypergeometric
        distribution for each cell x guide pair.

        Parameters
        ----------
        x : np.ndarray
            vector of guide counts for a single cell
        idx : int
            index of the cell in the matrix
        """
        return hypergeom.sf(x, self.population, self.successes, self.draws[idx])

    def _adjust_pvalues(self, pvalues: np.ndarray):
        """
        Adjust pvalues using the Bonferroni correction
        """
        adj_pvalues = adjust(pvalues, self.method)
        adj_pvalues = np.clip(adj_pvalues, np.min(adj_pvalues[adj_pvalues != 0]), 1)
        return adj_pvalues.reshape(pvalues.shape)

    def test(self):
        """
        Performs cell x guide geometric testing
        """
        with Pool(self.n_jobs) as p:
            pv_mat = np.vstack(
                p.starmap(
                    self._hypergeometric_test,
                    zip(self.matrix, np.arange(self._n_cells)),
                )
            )

        pv_mat = np.clip(pv_mat, np.min(pv_mat[pv_mat != 0]), 1)
        self.pv_mat = self._adjust_pvalues(pv_mat)
        self.is_fit = True

    def _generate_assignments(self, threshold=0.05):
        if not self.is_fit:
            AttributeError("Please run `.test()` method first")
        self._assignment_matrix = self.pv_mat < threshold
        self._is_assigned = True

    def _calculate_log_odds(self, lor_threshold: float):
        """
        calculates log-odds of each significant guide compared to the next
        most insignificant guide
        """
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")

        # instantiate the log odds matrix
        self.lor_matrix = np.zeros((self._n_cells, self._n_guides))

        # calculate the log odds for each cell
        for i in np.arange(self._n_cells):

            sig_idx = np.flatnonzero(self._assignment_matrix[i])
            sig_idx = sig_idx[np.argsort(self.pv_mat[i, sig_idx])]

            for j in sig_idx:
                min_insig = self.pv_mat[i][~self._assignment_matrix[i]].min()
                lor = logit(min_insig) - logit(self.pv_mat[i, j])
                if lor > lor_threshold:
                    self.lor_matrix[i, j] = lor
                else:
                    self._assignment_matrix[i, j] = False

        self._is_lor_calculated = True

    def _generate_labels(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        guide_indices = np.arange(self._m_total)
        guide_mask = guide_indices[self.passing_guides]
        self.labels = [
            self.guide_names[guide_mask[np.flatnonzero(self._assignment_matrix[i])]]
            for i in np.arange(self._n_cells)
        ]

    def _select_counts(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        self.counts = [
            self.matrix[i][np.flatnonzero(self._assignment_matrix[i])]
            for i in np.arange(self._n_cells)
        ]

    def _select_pvalues(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        self.pvalues = [
            self.pv_mat[i][np.flatnonzero(self._assignment_matrix[i])]
            for i in np.arange(self._n_cells)
        ]

    def _select_log_odds(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        self.log_odds = [
            self.lor_matrix[i][np.flatnonzero(self._assignment_matrix[i])]
            for i in np.arange(self._n_cells)
        ]

    def _calculate_moi(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        self.moi = np.sum(self._assignment_matrix, axis=1)

    def _filter_significant(
        self,
        pvalue_threshold: float = 0.05,
        lor_threshold: float = 10.0,
    ):
        """
        Calculates the MOI for each cell

        Parameters
        ----------
        pvalue_threshold : float
            pvalue threshold for significance (used on the adjusted pvalues)
        lor_threshold : float
            log odds ratio threshold for significance
        """
        self._generate_assignments(pvalue_threshold)
        self._calculate_log_odds(lor_threshold)
        self._generate_labels()
        self._select_counts()
        self._select_pvalues()
        self._select_log_odds()
        self._calculate_moi()

    def _assemble_dataframe(self):
        """
        Assemble the assignment results into a dataframe

        Returns
        -------
        df : pd.DataFrame
            dataframe with assignment results and calculated metrics
        """
        cell_id_in = np.arange(self._n_total)[self.passing_cells]
        cell_id_out = np.arange(self._n_total)[~self.passing_cells]

        cell_name_in = self.cell_names[self.passing_cells]
        cell_name_out = self.cell_names[~self.passing_cells]

        frame = pd.DataFrame(
            {
                "cell_id": cell_name_in,
                "assignment": self.labels,
                "counts": self.counts,
                "pvalues": self.pvalues,
                "log_odds": self.log_odds,
                "moi": self.moi,
                "n_umi": self.draws,
                "min_pvalue": self.pv_mat.min(axis=1),
                "max_count": self.matrix.max(axis=1),
                "tested": True,
            },
            index=cell_id_in,
        )
        null = pd.DataFrame(
            {
                "cell_id": cell_name_out,
                "assignment": [
                    np.array([]) for _ in np.arange(np.sum(~self.passing_cells))
                ],
                "counts": [
                    np.array([]) for _ in np.arange(np.sum(~self.passing_cells))
                ],
                "pvalues": [
                    np.array([]) for _ in np.arange(np.sum(~self.passing_cells))
                ],
                "log_odds": [
                    np.array([]) for _ in np.arange(np.sum(~self.passing_cells))
                ],
                "moi": np.nan,
                "n_umi": np.nan,
                "min_pvalue": np.nan,
                "max_count": np.nan,
                "log_odds": np.nan,
                "tested": False,
            },
            index=cell_id_out,
        )
        return pd.concat([frame, null]).sort_index()


    def assignments(
            self, 
            pvalue_threshold: float = 0.05, 
            lor_threshold: float = 10.0
        ) -> pd.DataFrame:
        """
        Returns a dataframe for all assignments with significance thresholds

        Parameters
        ----------
        pvalue_threshold : float
            pvalue threshold for significance (used on the adjusted pvalues)
        lor_threshold : float
            log odds ratio threshold for significance
        """
        self._filter_significant(pvalue_threshold, lor_threshold)
        return self._assemble_dataframe()

