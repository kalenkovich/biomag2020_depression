import os
import shutil
from pathlib import Path

from bids import BIDSLayout

project_root = Path()
original_data_dir = project_root / 'BIOMAG2020_comp_data'
bids_root = project_root / 'data_bids'

layout = BIDSLayout(bids_root, validate=True)

# TODO: get subjects and sessions from json files

subjects = layout.get_subjects()
sessions = layout.get_sessions()

test_pipeline_dir = bids_root / 'derivatives' / 'test_pipeline'

rule all:
    input:
        expand(os.path.join(test_pipeline_dir, 'sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg.json'), subject=subjects, session=sessions)

rule copy_json:
    input:
        os.path.join(bids_root, 'sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg.json')
    output:
        os.path.join(test_pipeline_dir, 'sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg.json')
    run:
         shutil.copy(input[0], output[0])
