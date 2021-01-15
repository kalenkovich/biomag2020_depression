import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import papermill as pm
from bids import BIDSLayout
from nbconvert import HTMLExporter

project_root = Path() / '..'
original_data_dir = project_root / 'BIOMAG2020_comp_data'
bids_root = Path(os.environ['biomag2020_data-bids'])

layout = BIDSLayout(bids_root, validate=True)

json_files = layout.get(suffix='meg', extension='json')

# Lists for session-level rules
subjects = [json_file.get_entities()['subject'] for json_file in json_files]
sessions = [json_file.get_entities()['session'] for json_file in json_files]

# Create lists for subject-level rules
subjects_df = (
    pd.DataFrame(dict(subject=subjects, session_id=sessions))
    .sort_values(by=['subject', 'session_id'])  # otherwise, there is no definitive session 1 and session 2)
)
subjects_df['session_number'] = subjects_df.groupby('subject').cumcount() + 1
# subjects_df looks like this:
#     subject  session_id  session_number
# 0  BQBBKEBX  1457629800               1
# 1  BQBBKEBX  1458832200               2
# 2  BYADLMJH  1416503760               1
# 3  BYADLMJH  1417706220               2


test_pipeline_dir = bids_root / 'derivatives' / 'test_pipeline'
template = os.path.join('sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg')
preprocessing_report_template = os.path.join(test_pipeline_dir, 'sub-{subject}_task-restingstate')

def find_ics(raw, ica, verbose=False):
    heart_ics, _ = ica.find_bads_ecg(raw, verbose=verbose)
    horizontal_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MLF14-1609', verbose=verbose)
    vertical_eye_ics, _ = ica.find_bads_eog(raw, ch_name='MLF21-1609', verbose=verbose)

    all_ics = heart_ics + horizontal_eye_ics + vertical_eye_ics
    # find_bads_E*G returns list of np.int64, not int
    all_ics = map(int, all_ics)
    # Remove duplicates.
    all_ics = list(set(all_ics))

    return all_ics

def find_ics_iteratively(raw, ica, verbose=False):
    ics = []

    new_ics = True  # so that the while loop initiates at all
    while new_ics:
        raw_copy = raw.copy()

        # Remove all components we've found so far
        ica.exclude = ics
        ica.apply(raw_copy)
        # Identify bad components in cleaned data
        new_ics = find_ics(raw_copy, ica, verbose=verbose)

        ics += new_ics

    return ics

rule all:
    input:
         expand(os.path.join(test_pipeline_dir, template+'_PSD_raw.png'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template+'_PSD_linearly_filtered.png'), zip, subject=subjects,
                session=sessions),
         expand(os.path.join(test_pipeline_dir, template + '-ics-removed.fif'), zip, subject=subjects,
                session=sessions),
         expand(preprocessing_report_template + '_preproc_report.html', zip, subject=np.unique(subjects)),

rule linear_filtering:
    input:
        os.path.join(bids_root, template+'.json')
    output:
        os.path.join(test_pipeline_dir, template+'.fif')
    run:
        raw_path = Path(input[0]).with_suffix('.ds')
        raw: mne.io.ctf.ctf.RawCTF = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)
        raw.resample(300, npad='auto')
        raw.notch_filter([60, 120],
                 filter_length='auto',
                 phase='zero')
        raw.filter(0.3, None, fir_design='firwin')
        raw.save(output[0])

rule draw_raw_psd:
    input:
        os.path.join(bids_root, template+'.json')
    output:
        os.path.join(test_pipeline_dir, template+'_PSD_raw.png')
    run:
        raw_path = Path(input[0]).with_suffix('.ds')
        raw = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)
        p = raw.plot_psd(show=False, fmax=150)
        p.axes[0].set_ylim(-30, 60)
        p.savefig(output[0])

rule draw_psd_linearly_filtered:
    input:
        os.path.join(test_pipeline_dir, template+'.fif')
    output:
        os.path.join(test_pipeline_dir, template+'_PSD_linearly_filtered.png')
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        p = raw.plot_psd(show=False, fmax=150)
        p.axes[0].set_ylim(-30, 60)
        p.savefig(output[0])

rule fit_ica:
    input:
        os.path.join(test_pipeline_dir, template+'.fif')
    output:
        os.path.join(test_pipeline_dir, template+'.ica')
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        raw.filter(1, None)
        ica = mne.preprocessing.ICA(random_state=2, n_components=25, verbose=False)
        ica.fit(raw)
        ica.save(output[0])

rule find_ics:
    input:
        os.path.join(test_pipeline_dir, template+'.fif'),
        os.path.join(test_pipeline_dir, template+'.ica')
    output:
        os.path.join(test_pipeline_dir, template+'.ics.pickle'),
        os.path.join(test_pipeline_dir, template+'_ics_properties.pickle')
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(input[1])
        ics = find_ics_iteratively(raw, ica, verbose=False)
        pd.to_pickle(ics, output[0])
        # When ics is empty, plot all components (`ics or None` will evaluate to None)
        component_figures = ica.plot_properties(raw, picks=(ics or None), show=False)
        pd.to_pickle(component_figures, output[1])

rule remove_artifactual_ics:
    input:
        os.path.join(test_pipeline_dir, template + '.fif'),
        os.path.join(test_pipeline_dir, template+'.ica'),
        os.path.join(test_pipeline_dir, template+'.ics.pickle')
    output:
        os.path.join(test_pipeline_dir, template + '-ics-removed.fif')
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(input[1])
        ics = pd.read_pickle(input[2])
        ica.exclude = ics
        raw_ics_removed = ica.apply(raw)
        raw_ics_removed.save(output[0])


def inputs_for_report(wildcards):
    subject = wildcards.subject

    session1 = subjects_df.query('subject == @subject and session_number == 1').session_id.values[0]
    session2 = subjects_df.query('subject == @subject and session_number == 2').session_id.values[0]

    prefix1 = os.path.join(test_pipeline_dir, template.format(subject=subject, session=session1))
    prefix2 = os.path.join(test_pipeline_dir, template.format(subject=subject, session=session2))

    return dict(
        filtered_data_1 = prefix1 + '.fif',
        psd_image_before_1 = prefix1 + '_PSD_raw.png',
        psd_image_after_1 = prefix1 + '_PSD_linearly_filtered.png',

        filtered_data_2 = prefix2 + '.fif',
        psd_image_before_2 = prefix2 + '_PSD_raw.png',
        psd_image_after_2 = prefix2 + '_PSD_linearly_filtered.png',

        ica_object_1 = prefix1 + '.ica',
        ica_bad_ics_1 = prefix1 + '.ics.pickle',
        ica_figures_1 = prefix1 + '_ics_properties.pickle',

        ica_object_2 = prefix2 + '.ica',
        ica_bad_ics_2 = prefix2 + '.ics.pickle',
        ica_figures_2 = prefix2 + '_ics_properties.pickle'
    )


rule make_preproc_report:
    input:
        unpack(inputs_for_report)
    output:
        preprocessing_report_template + '_preproc_report.html'
    run:
        # create ipynb
        ipynb_path = Path(output[0]).with_suffix('.ipynb')
        _ = pm.execute_notebook(
            '01_preprocessing\\report.ipynb',
            ipynb_path,
            parameters=input
        )

        # convert to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        body, resources = html_exporter.from_file(ipynb_path)

        with Path(output[0]).open('wt') as f:
            f.write(body)

        # remove ipynb
        os.remove(str(ipynb_path))
