#!/usr/bin/env python
# coding: utf-8

from pathlib import Path


import mne
import mne_bids


project_root = Path() / '..'
original_data_dir = project_root / 'BIOMAG2020_comp_data'
bids_root = project_root / 'data_bids'


raw = mne.io.read_raw_ctf(str(original_data_dir / 'BQBBKEBX_rest_20160310_01.ds'))
bids_basename = mne_bids.make_bids_basename(
    subject=raw.info['subject_info']['his_id'],
    session=str(raw.info['meas_id']['secs']),
    task='restingstate'
)
mne_bids.write_raw_bids(
    raw=raw, bids_basename=bids_basename, bids_root=bids_root)

