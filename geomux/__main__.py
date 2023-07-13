import sys
import argparse
from geomux import Geomux, read_table, read_anndata


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input table to assign"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        help="output table of barcode assignments (default=stdout)",
    )
    parser.add_argument(
        "-u",
        "--min_umi",
        type=int,
        required=False,
        default=5,
        help="minimum number of UMIs to consider a cell (default=5)",
    )
    parser.add_argument(
        "-c",
        "--min_cells",
        type=int,
        required=False,
        default=100,
        help="minimum number of cells to consider a guide (default=100)",
    )
    parser.add_argument(
        "-t",
        "--pvalue_threshold",
        type=float,
        required=False,
        default=0.05,
        help="Pvalue threshold to use after pvalue correction (default=0.05)",
    )
    parser.add_argument(
        "-T",
        "--lor_threshold",
        type=float,
        required=False,
        default=10.0,
        help="Log odds ratio threshold to use (default=10.0)",
    )
    parser.add_argument(
        "-C",
        "--correction",
        type=str,
        required=False,
        default="bh",
        help="Pvalue correction method to use (default=bh)",
    )
    parser.add_argument(
        "-j",
        "--n_jobs",
        type=int,
        required=False,
        default=1,
        help="Number of jobs to use when calculating hypergeometric distributions (default=1)",
    )
    args = parser.parse_args()
    return args


def main_cli():
    args = get_args()

    if args.input.endswith(".h5ad"):
        matrix = read_anndata(args.input)
    else:
        matrix = read_table(args.input)

    if args.correction not in ["bh", "bonferroni", "by"]:
        raise ValueError("Correction method must be one of: bh, bonferroni, by")

    gx = Geomux(
        matrix,
        cell_names=matrix.index.values,
        guide_names=matrix.columns.values,
        min_umi=args.min_umi,
        min_cells=args.min_cells,
        n_jobs=args.n_jobs,
        method=args.correction,
    )
    gx.test()
    assignments = gx.assignments(
        pvalue_threshold=args.pvalue_threshold,
        lor_threshold=args.lor_threshold,
    )

    if not args.output:
        assignments.to_csv(sys.stdout, sep="\t", index=False)
    else:
        assignments.to_csv(args.output, sep="\t", index=False)
