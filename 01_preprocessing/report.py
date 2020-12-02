#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os


from IPython.display import Image


import mne


# Folders

data_bids_root = Path(os.environ['biomag2020_data-bids'])
subject_root = data_bids_root / 'derivatives' / 'test_pipeline' / 'sub-BQBBKEBX'
session_1 = subject_root / 'ses-1457629800'
session_2 = subject_root / 'ses-1458832200'


# Files

filtered_data_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg.fif'
psd_image_before_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg_PSD_raw.png'
psd_image_after_1 = session_1 / 'meg' / 'sub-BQBBKEBX_ses-1457629800_task-restingstate_meg_PSD_linearly_filtered.png'

filtered_data_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg.fif'
psd_image_before_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg_PSD_raw.png'
psd_image_after_2 = session_2 / 'meg' / 'sub-BQBBKEBX_ses-1458832200_task-restingstate_meg_PSD_linearly_filtered.png'


info_1 = mne.io.read_raw_fif(filtered_data_1, verbose=False).info
info_2 = mne.io.read_raw_fif(filtered_data_2, verbose=False).info


# # Subject info

assert info_1['subject_info']['his_id'] == info_2['subject_info']['his_id'] 
subject_id = info_1['subject_info']['his_id']

session_id_1 = info_1['meas_id']['secs']
session_id_2 = info_2['meas_id']['secs']


print(f'''subject ID: {subject_id}
two sessions: {session_id_1}, {session_id_2}
''')


# # Linear filtering

assert info_1['sfreq'] == info_2['sfreq']
sfreq = info_1['sfreq']

assert info_1['highpass'] == info_2['highpass']
highpass = round(info_1['highpass'], 2)

assert info_1['lowpass'] == info_2['lowpass']
lowpass = info_1['lowpass']


print(f'''Parameters:
sampling rate after resampling: {sfreq}
highpass: {highpass}
lowpass: {lowpass}
notch filtering: not saved automatically
''')


# ## Session 1

# PSD before filering:

Image(psd_image_before_1)


# PSD after filtering:

Image(psd_image_after_1)


# ## Session 2

# PSD before filering:

Image(psd_image_before_2)


# PSD after filtering:

Image(psd_image_after_2)


# # ICA
