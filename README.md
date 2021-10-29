# geomux

A tool that assigns guides to cell barcodes. 

Uses a hypergeometric distribution to calculate the pvalue of observing the specific count of a guide for each guide in each barcode. Then it calculates the log odds ratio between the most significant and second most significant pvalue which measures the ratio of observing a doublet (double infection). Then a threshold is applied and the cells that pass are assigned the guide of their most significant expression

# Installation
```bash
git clone https://github.com/noamteyssier/geomux
cd geomux
conda env create --file=env.yaml
pip install -e .
```

# Usage
Geomux can be used either as a commandline tool or as a python module

## Commandline
when pip installing, an executable will be placed in your conda bin path. So you can call it directly from wherever in your filesystem
### Usage
```bash
# example usage
geomux -i <input.tab> -o <output.tab>
```
### Help Menu
```txt
$ geomux --help

usage: geomux [-h] -i INPUT [-o OUTPUT] [-u MIN_UMI] [-l MIN_LOR] [-s SCALAR] [-j N_JOBS]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input table to assign
  -o OUTPUT, --output OUTPUT
                        output table of barcode assignments (default=stdout)
  -u MIN_UMI, --min_umi MIN_UMI
                        minimum number of UMIs to consider a cell (default=5)
  -l MIN_LOR, --min_lor MIN_LOR
                        Log2 odds ratio threshold (default=1.0)
  -s SCALAR, --scalar SCALAR
                        scalar to use to avoid zeroes in log2 odds ratio calculation (default=0)
  -j N_JOBS, --n_jobs N_JOBS
                        Number of jobs to use when calculating hypergeometric distributions (default=1)
```

## Python Module
You can check out the example notebook in `example/GeomuxJup.ipynb` but briefly:

```python
from geomux import Geomux, read_table

input = "filename.tab"

frame = read_table(input)
geom = Geomux()
geom.fit(frame)
geom.predict()
assignments = geom.assignments()
```
