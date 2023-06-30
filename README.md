# geomux

A tool that assigns guides to cell barcodes. 

Uses a hypergeometric distribution to calculate the pvalue of observing the
specific count of a guide for each guide in each barcode.
This can be used to calculate the MOI of the cell and assigned guides for each cell.
The resulting dataframe can then be used to intersect with your original data
to assign every cell to a barcode and allows you to filter
for the MOI you're interested in working with.

## Installation

```bash
pip install geomux
```

## Usage

Geomux can be used either as a commandline tool or as a python module

### Commandline

when pip installing, an executable will be placed in your bin path. So you can call it directly from wherever in your filesystem

```bash
# example usage
geomux -i <input.tab / input.h5ad> -o <output.tsv>
```

You can also run the help flag to see the help menu for parameter options.

```txt
$ geomux --help

usage: geomux [-h] -i INPUT [-o OUTPUT] [-u MIN_UMI] [-t THRESHOLD] [-c CORRECTION] [-j N_JOBS] [-q]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input table to assign
  -o OUTPUT, --output OUTPUT
                        output table of barcode assignments (default=stdout)
  -u MIN_UMI, --min_umi MIN_UMI
                        minimum number of UMIs to consider a cell (default=5)
  -c MIN_CELLS, --min_cells MIN_CELLS
                        minimum number of cells to consider a guide (default=100)
  -t THRESHOLD, --threshold THRESHOLD
                        Pvalue threshold to use after pvalue correction (default=0.05)
  -C CORRECTION, --correction CORRECTION
                        Pvalue correction method to use (default=bh)
  -j N_JOBS, --n_jobs N_JOBS
                        Number of jobs to use when calculating hypergeometric distributions (default=1)
  -q, --quiet           Suppress progress messages
```

### Python Module

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

#### Processing an h5ad file format

```python
from geomux import Geomux, read_anndata

input = "filename.h5ad"

matrix = read_anndata(input)
gx = Geomux(
    matrix,
    cell_names=matrix.index.values,
    guide_names=matrix.columns.values,
)
gx.test()
assignments = gx.assignments
```

## Outputs

The results of `geomux` will be an assignment dataframe that has as many
observations as there are input cells.

The columns of this dataframe will include:

- cell_id
    - The name of the cell provided or the index.
- assignment
    - A list representing all significant guides within that cell.
- moi
    - The number of significant guides within the cell.
- n_umi
    - The number of UMIs observed in the cell.
- p_value
    - The adjusted p-value of the hypergeometric test for that cell/guide test.
- log_odds
    - The log odds of observing the highest scoring guide compared to the second highest.
- tested
    - A bool flag representing whether the cell was included in the test (or `False` if it was filtered for low UMI counts)
