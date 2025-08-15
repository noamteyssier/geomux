import logging
from multiprocessing import Pool
from typing import List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, csr_matrix
from scipy.special import logit
from scipy.stats import false_discovery_control, hypergeom

# Sets the maximum probability for p=1 when measuring log-odds
MAX_PROB = 1 - 1e-10
BACKUP_DELIMITER = "::"


class Geomux:
    def __init__(
        self,
        matrix: Union[np.ndarray, pd.DataFrame, ad.AnnData],
        cell_names: Optional[Union[List[str], np.ndarray]] = None,
        guide_names: Optional[Union[List[str], np.ndarray]] = None,
        min_umi: int = 5,
        min_cells: int = 100,
        n_jobs: int = 4,
        delimiter: str = "|",
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
        delimiter: str
            delimiter used to separate multiple values in output table
        """

        # Load the matrix
        if isinstance(matrix, pd.DataFrame):
            self.matrix = matrix.values
        elif isinstance(matrix, ad.AnnData):
            if matrix.X is None:
                raise ValueError("AnnData object must have a .X attribute")
            if isinstance(matrix.X, np.ndarray) or isinstance(matrix.X, np.matrix):
                self.matrix = np.array(matrix.X)
            elif isinstance(matrix.X, csr_matrix) or isinstance(matrix.X, csc_matrix):
                self.matrix = np.array(matrix.X.todense())
            else:
                raise ValueError(
                    "AnnData object must have a numpy array or sparse matrix as .X attribute"
                )
        else:
            self.matrix = matrix

        # Load the cell and guide names
        if cell_names is None:
            if isinstance(matrix, ad.AnnData):
                cell_names = np.array(matrix.obs_names)
            else:
                cell_names = np.arange(matrix.shape[0])
        else:
            assert len(cell_names) == matrix.shape[0]  # type: ignore
            cell_names = np.array(cell_names)

        if guide_names is None:
            if isinstance(matrix, ad.AnnData):
                guide_names = np.array(matrix.var_names)
            else:
                guide_names = np.arange(matrix.shape[1])
        else:
            assert len(guide_names) == matrix.shape[1]  # type: ignore
            guide_names = np.array(guide_names)

        self.cell_names = cell_names
        self.guide_names = guide_names

        # Set the parameters
        self.min_umi = min_umi
        self.min_cells = min_cells
        self.n_jobs = n_jobs
        self._n_total = matrix.shape[0]
        self._m_total = matrix.shape[1]
        self.delimiter = delimiter

        self._filter_matrix()
        self._validate_guide_names()
        self._fit_parameters()

        self._n_cells = self.matrix.shape[0]
        self._n_guides = self.matrix.shape[1]
        self._n_tests = self._n_cells * self._n_guides

        self.is_fit = False
        self.labels = []

    def _filter_matrix(self):
        """
        Filters the matrix to only include cells with at least
        `min_umi` UMIs
        """
        logging.info("--- Filtering matrix ---")
        logging.info(f"Original matrix shape: {self.matrix.shape}")

        cell_sums = self.matrix.sum(axis=1).flatten()
        guide_sums = self.matrix.sum(axis=0).flatten()
        self.passing_cells = cell_sums >= self.min_umi
        self.passing_guides = guide_sums >= self.min_cells
        self.matrix = self.matrix[self.passing_cells][:, self.passing_guides]

        # Stored for later use
        self._filtered_counts = cell_sums[~self.passing_cells].copy()

        logging.info(f"Filtered matrix shape: {self.matrix.shape}")
        logging.info("")

        logging.info("--- Summary statistics ---")
        logging.info(f"Average number of UMIs per cell: {cell_sums.mean():.2f}")
        logging.info(f"Variance of UMIs per cell: {cell_sums.var():.2f}")
        logging.info(
            f"Cell UMI counts range from {cell_sums.min()} to {cell_sums.max()}"
        )

        logging.info(f"Average number of cells per guide: {guide_sums.mean():.2f}")
        logging.info(f"Variance of cells per guide: {guide_sums.var():.2f}")
        logging.info(
            f"Guide cell counts range from {guide_sums.min()} to {guide_sums.max()}"
        )
        logging.info("")

        old_cell_size = self._n_total
        new_cell_size = self.matrix.shape[0]

        logging.info("--- Filtering Statistics ---")
        logging.info(
            "Removed {} ({:.2f}%) cells with < {} UMIs".format(
                old_cell_size - new_cell_size,
                100 * (old_cell_size - new_cell_size) / old_cell_size,
                self.min_umi,
            )
        )

        old_guide_size = self._m_total
        new_guide_size = self.matrix.shape[1]
        logging.info(
            "Removed {} ({:.2f}%) guides with < {} Cells".format(
                old_guide_size - new_guide_size,
                100 * (old_guide_size - new_guide_size) / old_guide_size,
                self.min_cells,
            )
        )
        logging.info("")

        if self.matrix.shape[0] == 0:
            raise ValueError(
                "No cells passed the UMI threshold. Try lowering the min_umi parameter"
            )
        if self.matrix.shape[1] == 0:
            raise ValueError(
                "No guides passed the cell threshold. Try lowering the min_cells parameter"
            )

    def _validate_guide_names(self):
        any_conflicts = False
        for g in self.guide_names:
            if self.delimiter in str(g):
                if BACKUP_DELIMITER in str(g):
                    raise ValueError(
                        f"Guide: {g} contains restricted characters {self.delimiter} or {BACKUP_DELIMITER}. Please update delimiter to use a character that is not found in your guide names"
                    )
                any_conflicts = True
        if any_conflicts:
            logging.warning(
                f"Found a conflicting guide name with `{self.delimiter}`. Updating to delimiter: `{BACKUP_DELIMITER}`"
            )
            self.delimiter = BACKUP_DELIMITER

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
        return hypergeom.sf(x - 1, self.population, self.successes, self.draws[idx])

    def _adjust_pvalues(self, pvalues: np.ndarray):
        """
        Adjust pvalues using the Bonferroni correction
        """
        adj_pvalues = false_discovery_control(pvalues, method="bh")
        adj_pvalues = np.clip(adj_pvalues, np.min(adj_pvalues[adj_pvalues != 0]), 1)
        return adj_pvalues.reshape(pvalues.shape)

    def test(self):
        """
        Performs cell x guide geometric testing
        """
        logging.info("--- Hypergeometric Testing ---")
        logging.info(f"Number of cells to test: {self._n_cells}")
        with Pool(self.n_jobs) as p:
            pv_mat = np.vstack(
                p.starmap(
                    self._hypergeometric_test,
                    zip(self.matrix, np.arange(self._n_cells)),
                )
            )
        logging.info("")

        pv_mat = np.clip(pv_mat, np.min(pv_mat[pv_mat != 0]), 1)

        logging.info("--- P-value Adjustment ---")
        self.pv_mat = self._adjust_pvalues(pv_mat)
        logging.info("")

        self.is_fit = True

    def _generate_assignments(self, threshold=0.05):
        if not self.is_fit:
            AttributeError("Please run `.test()` method first")

        logging.info("--- P-Value Thresholding ---")
        self._assignment_matrix = self.pv_mat < threshold

        self._n_assigned = np.sum(self._assignment_matrix.sum(axis=1) > 0)
        logging.info(f"{self._n_assigned} cells assigned to guides")
        logging.info("")

        self._is_assigned = True

    def _calculate_log_odds(self, lor_threshold: float):
        """
        calculates log-odds of each significant guide compared to the next
        most insignificant guide

        This is done in a dynamic pattern, where the most significant guide
        is compared to the next-most insignificant guide. If the log-odds
        is greater than the threshold, the guide is kept. Otherwise, the
        guide is removed from the assignment matrix and it will be included
        in the next comparison.

        Parameters
        ----------
        lor_threshold : float
            log-odds threshold for guide inclusion (LOR must be greater than this)
        """
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")

        logging.info("--- Log Odds Calculation ---")

        # instantiate the log odds matrix
        self.lor_matrix = np.zeros((self._n_cells, self._n_guides))

        # calculate the log odds for each cell
        for i in np.arange(self._n_cells):
            # select the significant guides
            sig_idx = np.flatnonzero(self._assignment_matrix[i])

            # Sort the indices by pvalue (descending)
            sort_idx = np.argsort(self.pv_mat[i, sig_idx])[::-1]

            # sort the significant guides by pvalue (descending)
            sig_idx = sig_idx[sort_idx]

            for j in sig_idx:
                # find the next most significant insignificant guide
                min_insig = self.pv_mat[i][~self._assignment_matrix[i]].min()

                # sets the max probability in case of p=1
                min_insig = min(min_insig, MAX_PROB)

                # calculate the log odds
                lor = logit(min_insig) - logit(self.pv_mat[i, j])

                # if the log odds is greater than the threshold, keep the guide
                # otherwise, remove the guide from the assignment matrix
                if lor > lor_threshold:
                    self.lor_matrix[i, j] = lor
                else:
                    self._assignment_matrix[i, j] = False

        self._n_assigned = np.sum(self._assignment_matrix.sum(axis=1) > 0)
        logging.info(f"{self._n_assigned} cells assigned to guides")
        logging.info("")
        self._is_lor_calculated = True

    def _generate_labels(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        guide_indices = np.arange(self._m_total)
        guide_mask = guide_indices[self.passing_guides]

        self.labels = []
        for i in np.arange(self._n_cells):
            assignment_indices = np.flatnonzero(self._assignment_matrix[i])
            guide_names = self.delimiter.join(
                [str(x) for x in self.guide_names[guide_mask[assignment_indices]]]
            )
            self.labels.append(guide_names)

    def _select_counts(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        self.counts = []
        for i in np.arange(self._n_cells):
            assignment_indices = np.flatnonzero(self._assignment_matrix[i])
            counts = self.delimiter.join(
                [str(int(x)) for x in self.matrix[i][assignment_indices]]
            )
            self.counts.append(counts)

    def _select_pvalues(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        self.pvalues = []
        for i in np.arange(self._n_cells):
            assignment_indices = np.flatnonzero(self._assignment_matrix[i])
            pvalues = self.delimiter.join(
                [str(x) for x in self.pv_mat[i][assignment_indices]]
            )
            self.pvalues.append(pvalues)

    def _select_log_odds(self):
        if not self._is_assigned:
            raise AttributeError("Must assign guides first")
        if not self._is_lor_calculated:
            raise AttributeError("Must calculate log odds first")

        self.log_odds = []
        for i in np.arange(self._n_cells):
            assignment_indices = np.flatnonzero(self._assignment_matrix[i])
            log_odds = self.delimiter.join(
                [str(x) for x in self.lor_matrix[i][assignment_indices]]
            )
            self.log_odds.append(log_odds)

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

        cell_name_in = self.cell_names[self.passing_cells]  # type: ignore
        cell_name_out = self.cell_names[~self.passing_cells]  # type: ignore

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
                "assignment": "",
                "counts": "",
                "pvalues": "",
                "log_odds": "",
                "moi": 0,
                "n_umi": self._filtered_counts,
                "min_pvalue": np.nan,
                "max_count": np.nan,
                "tested": False,
            },
            index=cell_id_out,
        )
        return pd.concat([frame, null.astype(frame.dtypes)]).sort_index()

    def assignments(
        self, pvalue_threshold: float = 0.05, lor_threshold: float = 10.0
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
