#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os


import papermill as pm


data_bids_root = Path(os.environ['biomag2020_data-bids'])
subject_root = data_bids_root / 'derivatives' / 'test_pipeline' / 'sub-BQBBKEBX'
session_1 = subject_root / 'ses-1457629800'
session_2 = subject_root / 'ses-1458832200'


parameters=dict(
    filtered_data_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg.fif',
    psd_image_before_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg_PSD_raw.png',
    psd_image_after_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg_PSD_linearly_filtered.png',

    filtered_data_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg.fif',
    psd_image_before_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg_PSD_raw.png',
    psd_image_after_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg_PSD_linearly_filtered.png',

    ica_object_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg.ica',
    ica_bad_ics_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg.ics.pickle',
    ica_figures_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg_ics_properties.pickle',

    ica_object_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg.ica',
    ica_bad_ics_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg.ics.pickle',
    ica_figures_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg_ics_properties.pickle', 
)

for key in parameters:
    parameters[key] = str(parameters[key])


output_notebook = 'report_executed.ipynb'


_ = pm.execute_notebook(
    'report.ipynb',
    output_notebook,
    parameters=parameters
)


import nbconvert
from nbconvert import HTMLExporter


html_exporter = HTMLExporter()
html_exporter.template_name = 'classic'


body, resources = html_exporter.from_file(output_notebook)


with Path(output_notebook).with_suffix('.html').open('wt') as f:
    f.write(body)

