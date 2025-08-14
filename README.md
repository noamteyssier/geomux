# geomux

A tool that assigns guides to cell barcodes.

Uses a hypergeometric distribution to calculate the pvalue of observing the
specific count of a guide for each guide in each barcode.
This can be used to calculate the MOI of the cell and assigned guides for each cell.
The resulting dataframe can then be used to intersect with your original data
to assign every cell to a barcode and allows you to filter
for the MOI you're interested in working with.

## Installation

`geomux` is distributed via [`uv`](https://docs.astral.sh/uv/)

```bash
uv tool install geomux
geomux --help
```

## Usage

Geomux can be used either as a commandline tool or as a python module

### Commandline

when installing via `uv`, an executable will be placed in your bin path. So you can call it directly from wherever in your filesystem

```bash
# example usage
geomux <input.tab / input.h5ad>
```

You can also run the help flag to see the help menu for parameter options.

```txt
$ geomux --help

Usage: geomux [OPTIONS] INPUT [OUTPUT]

╭─ Arguments ──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    input       TEXT      Input file path (tsv/h5ad) to assign guides. [default: None] [required]                   │
│      output      [OUTPUT]  Output file path (tsv) to save assignments. [default: geomux.tsv]                         │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --min-umi                 INTEGER  Minimum UMI count to consider a barcode [default: 5]                              │
│ --min-cells               INTEGER  Minimum number of barcodes to consider a guide [default: 100]                     │
│ --pvalue-threshold        FLOAT    Maximum pvalue (fdr) to consider a guide-assignment [default: 0.05]               │
│ --lor-threshold           FLOAT    Log odds ratio threshold to use [default: 10.0]                                   │
│ --correction              TEXT     Pvalue correction method to use [default: bh]                                     │
│ --n-jobs                  INTEGER  Number of jobs to use when calculating hypergeometric distributions [default: 1]  │
│ --help                             Show this message and exit.                                                       │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

### Python Module

#### Processing an h5ad file format

```python
import anndata as ad
from geomux import Geomux

input = "filename.h5ad"

adata = ad.read_h5ad(input)
gx = Geomux(adata)
gx.test()
assignments = gx.assignments
```

#### Processing a 3-column TSV of [barcode, guide, n_umi]

```python
from geomux import Geomux, read_table

input = "filename.tsv"

matrix = read_table(input)
gx = Geomux(
    matrix,
    cell_names=matrix.index.values,
    guide_names=matrix.columns.values,
)
gx.test()
assignments = gx.assignments()
```

## Outputs

The results of `geomux` will be an assignment dataframe that has as many
observations as there are input cells.

The columns of this dataframe will include:

| Column Name | Description |
| ----------- | ----------- |
| cell_id | The name of the cell provided or the index. |
| assignment | A list representing all significant guides within that cell. |
| counts | The number of UMIs observed in the cell. |
| pvalues | The adjusted p-value of the hypergeometric test for that cell/guide test. |
| log_odds | The log odds of observing the highest scoring guide compared to the second highest. |
| moi | The number of significant guides within the cell. |
| n_umi | The number of UMIs observed in the cell. |
| min_pvalue | The minimum pvalue across all significant guides within the cell. |
| max_count | The maximum count across all significant guides within the cell. |
| tested | A bool flag representing whether the cell was included in the test (or `False` if it was filtered for low UMI counts) |
