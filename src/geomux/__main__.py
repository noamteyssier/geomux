import anndata as ad
import numpy as np
import typer
from typing_extensions import Annotated

from geomux import Geomux, read_table


def main_cli(
    input: Annotated[
        str, typer.Argument(help="Input file path (tsv/h5ad) to assign guides.")
    ],
    output: Annotated[
        str, typer.Argument(help="Output file path (tsv) to save assignments.")
    ] = "geomux.tsv",
    min_umi: Annotated[
        int, typer.Option(help="Minimum UMI count to consider a barcode")
    ] = 5,
    min_cells: Annotated[
        int,
        typer.Option(help="Minimum number of barcodes to consider a guide"),
    ] = 100,
    pvalue_threshold: Annotated[
        float, typer.Option(help="Maximum pvalue (fdr) to consider a guide-assignment")
    ] = 0.05,
    lor_threshold: Annotated[
        float,
        typer.Option(
            help="Log odds ratio threshold to use (default=10.0)",
        ),
    ] = 10.0,
    correction: Annotated[
        str, typer.Option(help="Pvalue correction method to use (default=bh)")
    ] = "bh",
    n_jobs: Annotated[
        int,
        typer.Option(
            help="Number of jobs to use when calculating hypergeometric distributions (default=1)"
        ),
    ] = 1,
):
    # args = get_args()

    if input.endswith(".h5ad"):
        matrix = ad.read_h5ad(input)
        cell_names = np.array(matrix.obs_names.values)
        guide_names = np.array(matrix.var_names.values)
    else:
        matrix = read_table(input)
        cell_names = np.array(matrix.index.values)
        guide_names = np.array(matrix.columns.values)

    if correction not in ["bh", "bonferroni", "by"]:
        raise ValueError("Correction method must be one of: bh, bonferroni, by")

    gx = Geomux(
        matrix,
        cell_names=cell_names,
        guide_names=guide_names,
        min_umi=min_umi,
        min_cells=min_cells,
        n_jobs=n_jobs,
        method=correction,
    )
    gx.test()
    assignments = gx.assignments(
        pvalue_threshold=pvalue_threshold,
        lor_threshold=lor_threshold,
    )

    assignments.to_csv(output, sep="\t", index=False)


def main():
    typer.run(main_cli)
