# Preprocessing reports code

The actual preprocessing is done within the `Snakefile` file in the repository root.
In this folder, we have a template file `report.ipynb` for the `papermill` package to run a parameterized Jupyter notebook.
The way it works, `report.ipynb` has a single cell with all the parameters that change from subject to subject - these are paths to individual files the report is based on.
This cell is tagged `parameters` so that `papermill` knows which code should be substituted.
(use `View` -> `Cell Toolbar` -> `Tags` in Jupyter Notebook to see the tags).

The `papermill_code.ipynb` contains test code that runs this procedure for a single subject.
This code was later translated into a snakemake rule (`make_preproc_report`) that would be run for each subject.   

The `papermill` package outputs a Jupyter notebook file.
We then use `nbconvert` to convert it to HTML.
