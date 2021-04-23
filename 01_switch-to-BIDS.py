#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import mne
import mne_bids


# The assumed folder structure is a single project folder that contains
# the repo folder and the data folders 'BIOMAG2020_comp_data' with
# the original data and 'data_bids' with the BIDS data.
# 
# TODO: switch to using environment variables

project_root = Path() / '..'
original_data_dir = project_root / 'BIOMAG2020_comp_data'
bids_root = project_root / 'data_bids'


# # Convert to BIDS

# The ctf data is organized in folders whose name end with ".ds"
folders = list(original_data_dir.glob('*.ds'))
len(folders)


# We encountered a couple of files that mne couldn't read.
# 
# We just skipped over them manually by running the loop over all,
# letting it break with an error, then getting the index of the 
# offender (say, `n`) and then continuing processing the folders
# after it (looping over `folders[n + 1:]`).
# 
# TODO: make a list of the offending folders and skip them like adults.
# Otherwise, we can't re-run this script if we need to.

for folder in folders[59:]:
    raw = mne.io.read_raw_ctf(str(folder))
    bids_basename = mne_bids.make_bids_basename(
        subject=raw.info['subject_info']['his_id'],
        # There is a session number (1 or 2) in the folder name.
        # We didn't want to use
        session=str(raw.info['meas_id']['secs']),
        task='restingstate'
    )

    mne_bids.write_raw_bids(raw=raw, bids_basename=bids_basename, bids_root=bids_root, verbose=False, overwrite=True) #overwrite doesnt work


# When the loop above breaks, `folder` points to the recording we couldn't read.
# Below, we find its index and then restart the loop starting with the folder
# after the offender.

folders.index(folder)

