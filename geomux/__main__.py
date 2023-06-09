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
        "-l",
        "--min_lor",
        type=float,
        required=False,
        default=1.0,
        help="Log2 odds ratio threshold (default=1.0)",
    )
    parser.add_argument(
        "-s",
        "--scalar",
        type=int,
        required=False,
        default=1,
        help="scalar to use to avoid zeroes in log2 odds ratio calculation (default=0)",
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
        frame = read_anndata(args.input)
    else:
        frame = read_table(args.input)

    geom = Geomux(min_umi=args.min_umi, scalar=args.scalar, n_jobs=args.n_jobs)
    geom.fit(frame)
    geom.predict(min_lor=args.min_lor)
    assignments = geom.assignments()

    if not args.output:
        assignments.to_csv(sys.stdout, sep="\t", index=False)
    else:
        assignments.to_csv(args.output, sep="\t", index=False)
