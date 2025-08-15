import json
import logging

import anndata as ad
import numpy as np
import typer
from typing_extensions import Annotated

from geomux import Geomux, read_table
from geomux.utils import assignment_statistics


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
            help="Log odds ratio threshold to use",
        ),
    ] = 10.0,
    n_jobs: Annotated[
        int,
        typer.Option(
            help="Number of jobs to use when calculating hypergeometric distributions"
        ),
    ] = 1,
    delim: Annotated[
        str, typer.Option(help="Delimiter to use for multi-value columns in output")
    ] = "|",
    stats: Annotated[
        str | None,
        typer.Option(help="Output file to write assignment statistics to as json"),
    ] = None,
):
    if input.endswith(".h5ad"):
        matrix = ad.read_h5ad(input)
        cell_names = np.array(matrix.obs_names.values)
        guide_names = np.array(matrix.var_names.values)
    else:
        matrix = read_table(input)
        cell_names = np.array(matrix.index.values)
        guide_names = np.array(matrix.columns.values)

    gx = Geomux(
        matrix,
        cell_names=cell_names,
        guide_names=guide_names,
        min_umi=min_umi,
        min_cells=min_cells,
        n_jobs=n_jobs,
        delimiter=delim,
    )
    gx.test()
    assignments = gx.assignments(
        pvalue_threshold=pvalue_threshold,
        lor_threshold=lor_threshold,
    )

    logging.info(f"Writing assignments to file: {output}")
    assignments.to_csv(output, sep="\t", index=False)

    # Write the statistics dictionary as json
    if stats:
        logging.info(f"Writing assignment statistics to file: {stats}")
        statistics = assignment_statistics(assignments)
        with open(stats, "w+") as f:
            f.write(json.dumps(statistics, indent=2))


def _version():
    import sys

    if "--version" in sys.argv:
        from importlib.metadata import version
        import sys

        print(f"geomux {version('geomux')}")
        sys.exit(0)


def main():
    _version()
    typer.run(main_cli)
