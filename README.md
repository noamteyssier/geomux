# geomux

A tool that assigns guides to cell barcodes.

Uses a hypergeometric distribution to calculate the pvalue of observing the specific count of a guide for each guide in each barcode.
This can be used to calculate the MOI of the cell and assigned guides for each cell.
The resulting dataframe can then be used to intersect with your original data to assign every cell to a barcode and allows you to filter for the MOI you're interested in working with.

## Installation

`geomux` is distributed via [`uv`](https://docs.astral.sh/uv/)

```bash
uv tool install geomux
geomux --help
```

## Usage

Geomux can be used either as a commandline tool or as a python module

Geomux supports two modes of operation:

1. Hypergeometric testing
2. Gaussian Mixture Model testing

This can be set with the `--method` flag on the CLI or by using the relevant function (`geomux` or `gaussian_mixture`)

### Commandline

when installing via `uv`, an executable will be placed in your bin path. So you can call it directly from wherever in your filesystem

```bash
# example usage
geomux <input.tab / input.h5ad>
```

You can also run the help flag to see the help menu for parameter options.

```txt

 Usage: geomux [OPTIONS] INPUT [OUTPUT]

╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    input       TEXT      Input file path (tsv/h5ad) to assign guides. [required]                                                                 │
│      output      [OUTPUT]  Output file path (tsv) to save assignments. [default: geomux.tsv]                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --min-umi-cells                           INTEGER  Minimum UMI count to consider a barcode [default: 5]                                            │
│ --min-umi-guides                          INTEGER  Minimum number of barcodes to consider a guide [default: 5]                                     │
│ --fdr-threshold                           FLOAT    Maximum pvalue (fdr) to consider a guide-assignment [default: 0.05]                             │
│ --lor-threshold                           FLOAT    Log odds ratio threshold to use (None for adaptive thresholding)                                │
│ --adaptive-lor-scalar                     FLOAT    Scalar to adaptively set log odds ratio threshold                                               │
│ --subtract               --no-subtract             Subtract 1 from counts before testing. [default: subtract]                                      │
│ --stats                                   TEXT     Output file to write assignment statistics to as json                                           │
│ --method                                  TEXT     Method to use for assignment (geomux/mixture) [default: geomux]                                 │
│ --n-jobs                                  INTEGER  Number of jobs to use for parallel processing (mixture model only). -1 for all available cores. │
│                                                    [default: -1]                                                                                   │
│ --help                                             Show this message and exit.                                                                     │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Python Module

#### Processing an h5ad file format

```python
import anndata as ad
from geomux import geomux, gaussian_mixture

input = "filename.h5ad"

adata = ad.read_h5ad(input)

assignments_geomux = geomux(adata)
print(assignments_geomux)

assignments_mixture = gaussian_mixture(adata)
print(assignments_mixture)
```

#### Processing an gene x guide sparse matrix

```python
import anndata as ad
from geomux import geomux

input = "filename.h5ad"

adata = ad.read_h5ad(input)
matrix = adata.X.tocsr()
assignments = geomux(matrix)
print(assignments)
```

## Outputs

The results of `geomux` will be an assignment dataframe that has as many
observations as there are input cells.

The columns of this dataframe will include:

| Column Name        | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| cell_id            | The numerical index of this cell in the count matrix.                   |
| submatrix_id       | The numerical index of this cell in the filtered count matrix.          |
| cell               | The numerical index of this cell _or_ the name of the cell if provided. |
| moi                | The number of assigned guides for this cell.                            |
| n_umi              | The number of total UMIs observed in the cell.                          |
| assignment         | A '\|' separated string of the assigned guides for this cell.           |
| guide_ids_original | A '\|' separated string of the assigned guide numerical indices.        |
| umis               | A '\|' separated string of the assigned guide UMIs.                     |
| fdr                | A '\|' separated string of the false discovery rate of each assignment. |
| log_odds           | A '\|' separated string of the log-odds of each assignment.             |
| tested             | A bool designating whether this cell met the testing criteria.          |
