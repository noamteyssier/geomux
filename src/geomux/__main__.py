import json

import anndata as ad
import typer
from typing_extensions import Annotated

from geomux.mixture import gaussian_mixture

from . import geomux
from .utils import assignment_statistics


def main_cli(
    input: Annotated[
        str, typer.Argument(help="Input file path (tsv/h5ad) to assign guides.")
    ],
    output: Annotated[
        str, typer.Argument(help="Output file path (tsv) to save assignments.")
    ] = "geomux.tsv",
    min_umi_cells: Annotated[
        int, typer.Option(help="Minimum UMI count to consider a barcode")
    ] = 5,
    min_umi_guides: Annotated[
        int,
        typer.Option(help="Minimum number of barcodes to consider a guide"),
    ] = 5,
    fdr_threshold: Annotated[
        float, typer.Option(help="Maximum pvalue (fdr) to consider a guide-assignment")
    ] = 0.05,
    lor_threshold: Annotated[
        float | None,
        typer.Option(
            help="Log odds ratio threshold to use (None for adaptive thresholding)",
        ),
    ] = None,
    adaptive_lor_scalar: Annotated[
        float | None,
        typer.Option(
            help="Scalar to adaptively set log odds ratio threshold",
        ),
    ] = None,
    subtract: Annotated[
        bool, typer.Option(help="Subtract 1 from counts before testing.")
    ] = True,
    stats: Annotated[
        str | None,
        typer.Option(help="Output file to write assignment statistics to as json"),
    ] = None,
    method: Annotated[
        str, typer.Option(help="Method to use for assignment (geomux/mixture)")
    ] = "geomux",
    n_jobs: Annotated[
        int,
        typer.Option(
            help="Number of jobs to use for parallel processing (mixture model only). -1 for all available cores."
        ),
    ] = -1,
):
    adata = ad.read_h5ad(input)

    if method == "geomux":
        results = geomux(
            adata,
            min_umi_cells=min_umi_cells,
            min_umi_guides=min_umi_guides,
            fdr_threshold=fdr_threshold,
            lor_threshold=lor_threshold,
            adaptive_lor_scalar=adaptive_lor_scalar,
            subtract=subtract,
        )
        results.write_csv(output, separator="\t")
        if stats:
            statistics = assignment_statistics(results)
            with open(stats, "w+") as f:
                f.write(json.dumps(statistics, indent=2))
    elif method == "mixture":
        results = gaussian_mixture(
            adata,
            min_umi_cells=min_umi_cells,
            n_jobs=n_jobs,
        )
        results.write_csv(output, separator="\t")
        if stats:
            statistics = assignment_statistics(results)
            with open(stats, "w+") as f:
                f.write(json.dumps(statistics, indent=2))
    else:
        raise ValueError(f"Invalid method: {method}. Use `geomux` or `mixture`")


def _version():
    import sys

    if "--version" in sys.argv:
        import sys
        from importlib.metadata import version

        print(f"geomux {version('geomux')}")
        sys.exit(0)


def main():
    _version()
    typer.run(main_cli)
