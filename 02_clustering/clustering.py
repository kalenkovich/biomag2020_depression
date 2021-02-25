#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os

from bids import BIDSLayout
import numpy as np
import mne


bids_root = Path(os.environ['biomag2020_data-bids'])


layout = BIDSLayout(bids_root, validate=True, derivatives=True)
layout.derivatives


eigenvalue_files = layout.get(suffix='eigenvalues', extension='npy')


eigs_all = np.stack([np.load(f) for f in eigenvalue_files])

