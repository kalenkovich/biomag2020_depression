#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import shutil


import mne
import mne_bids
from bids import BIDSLayout


# # Move one file to BIDS 

project_root = Path() / '..'
original_data_dir = project_root / 'BIOMAG2020_comp_data'
bids_root = project_root / 'data_bids'


raw = mne.io.read_raw_ctf(str(original_data_dir / 'BQBBKEBX_rest_20160310_01.ds'))
bids_basename = mne_bids.make_bids_basename(
    subject=raw.info['subject_info']['his_id'],
    session=str(raw.info['meas_id']['secs']),
    task='restingstate'
)
# mne_bids.write_raw_bids(
#     raw=raw, bids_basename=bids_basename, bids_root=bids_root)


# # Save one derivative file 

input_filepath = bids_root / 'sub-BQBBKEBX' / 'ses-1457629800' / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg.json'


derivatives_root = bids_root / 'derivatives' / 'test_pipeline'


output_filepath = derivatives_root / input_filepath.relative_to(bids_root)


output_filepath.parent.mkdir(parents=True, exist_ok=True)


if not output_filepath.exists():
    shutil.copy(str(input_filepath), str(output_filepath))


# # Locate files using `pybids`

bids_layout = BIDSLayout(root=bids_root)


bids_layout.get()

