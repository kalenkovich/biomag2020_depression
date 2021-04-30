import os
from pathlib import Path

import bids
import mne
import numpy as np
import pandas as pd
import papermill as pm
import yaml
from bids import BIDSLayout
from nbconvert import HTMLExporter
from scipy.sparse import csgraph
from numpy import linalg

# TODO: switch to an environment variable for `original_data_dir`. Can just remove both it and `project_root`.
project_root = Path() / '..'
original_data_dir = project_root / 'BIOMAG2020_comp_data'
bids_root = Path(os.environ['biomag2020_data-bids'])

layout = BIDSLayout(bids_root, validate=True)

# ctf recording are folders, not single files. Therefore, we had to use some file as an anchor. We decided to use json
# file located in the folder that contains the '*.ds' folder with the recording itself. Snakemake can work with folders
# as input and outputs so we could have done that instead.
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
# ...

# Derivatives folders for the output
derivatives_dir = bids_root / 'derivatives'
preprocessing_pipeline_dir = derivatives_dir / '01_preprocessing'
eigenvalues_derivatives = derivatives_dir / '02_eigenvalues'

# Set up the templates for the inputs and outputs. Note the wildcards.

# This is the base template extended directly in the rule definitions.
template = os.path.join('sub-{subject}', 'ses-{session}', 'meg', 'sub-{subject}_ses-{session}_task-restingstate_meg')

# Later, we switched to defining templates here to make rules easier to read.
# TODO: switch all of them
preprocessing_report_template = os.path.join(preprocessing_pipeline_dir, 'sub-{subject}_task-restingstate')
manual_check_template = os.path.join(preprocessing_pipeline_dir, 'sub-{subject}_task-restingstate_manualCheck.yml')
eigenvalues_template = os.path.join(eigenvalues_derivatives, template + '-eigenvalues.npy')


def find_ics(raw, ica, verbose=False):
    """
    Finds ICA components associated with heart and eye artifacts.
    There was no ECG/EOG in the data, therefore we couldn't use them to search for artifacts.
    For heart artifacts, mne can do it without ECG.
    For eye artifacts, we manually selected two channels: one for the vertical and one for the horizontal ones.
    :param raw: the raw recording file
    :param ica: an mne.preprocessing.ICA with components already fitted
    :param verbose: verbose level of the `find_bads_ecg` and `find_bads_eog` methods of ica
    :return: list with indexes of heart and eye components
    """
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
    """
    Finds ICA components associated with heart and eye artifacts iteratively: find some, removes them, then find some
    more, removes them, etc.
    :param raw: the raw recording file
    :param ica: an mne.preprocessing.ICA with components already fitted
    :param verbose: verbose level of the `find_bads_ecg` and `find_bads_eog` methods of ica
    :return: list with indexes of components
    """
    ics = []

    new_ics = True  # so that the while loop initiates at all
    while new_ics:
        # We don't want to make any changes to the recording yet
        raw_copy = raw.copy()

        # Remove all components we've found so far
        ica.exclude = ics
        ica.apply(raw_copy)
        # Identify bad components in cleaned data
        new_ics = find_ics(raw_copy, ica, verbose=verbose)

        ics += new_ics

    return ics


def is_report_ok(check_result_path):
    """
    Checks the preprocessing report. The actual checking is done manually by the analysts. Once they checked the report,
    they add a yaml file with a single node `success` which can either be `yes` or `no`. For example:

    ```yaml
    success: yes
    ```

    :param check_result_path: path to the yaml file
    :return: bool - was the preprocessing deemed successful?
    """
    with open(check_result_path) as f:
        check_result = yaml.full_load(f)

    # There must be exactly one key
    assert list(check_result.keys()) == ['success'], '\n'.join([
        'Problem with {file}'.format(file=check_result_path),
        'Manual-check files must contain exactly one key - `success`'])
    # And it must be either True or False
    assert check_result['success'] is True or check_result['success'] is False, '\n'.join([
        'Problem with {file}'.format(file=check_result_path),
        '`success` can only be yes or no'])

    return check_result['success']


common_channels_path = os.path.join(preprocessing_pipeline_dir, 'common_channels.csv')


rule all:
    input:
        # expand(manual_check_template, subject=np.unique(subjects)),
        expand(os.path.join(preprocessing_pipeline_dir, template + '-ics-removed.fif'), zip, subject=subjects, session=sessions),
        expand(eigenvalues_template, zip, subject=subjects, session=sessions)
    # output:
    #     os.path.join(preprocessing_pipeline_dir, 'preprocessing-report_all.txt')
    # run:
    #     n_successes = 0
    #     problematic_subjects = list()
    #     for check_result_path in input:
    #         report_ok = is_report_ok(check_result_path)
    #         n_successes += report_ok
    #         if not report_ok:
    #             path_parsed = bids.layout.parse_file_entities(check_result_path)
    #             problematic_subjects.append(path_parsed['subject'])
    #
    #     with open(output[0], 'w') as f:
    #         f.write(f'Files from {len(input)} participants were processed.\n')
    #         f.write(f'From {n_successes} of them - successfully.\n')
    #         f.write('\n')
    #
    #         if len(problematic_subjects) > 0:
    #             f.write('Problematic subjects:\n')
    #             for problematic_subject in problematic_subjects:
    #                 f.write(problematic_subject + '\n')
    #         else:
    #             f.write('No problematic subjects.')


linearly_filtered_template = os.path.join(preprocessing_pipeline_dir, template + '.fif')
# NB: We haven't quite decided whether to define all the templates at the top or before it is first used.


rule linear_filtering:
    input:
        os.path.join(bids_root, template+'.json')
        # TODO: define a template
    output:
        linearly_filtered_template
    run:
        raw_path = Path(input[0]).with_suffix('.ds')
        raw: mne.io.ctf.ctf.RawCTF = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)
        raw.resample(300, npad='auto')
        raw.notch_filter([60, 120],  # higher harmonics are not present after the downsampling
                 filter_length='auto',  # the other two parameters were copied from a tutorial
                 phase='zero')
        raw.filter(0.3, None, fir_design='firwin')  # the `fir_desing` parameter was copied from a tutorial
        raw.save(output[0])
# TODO: name inputs and outputs


# Creates a pds plot of the raw data
rule draw_raw_psd:
    input:
        os.path.join(bids_root, template+'.json')
        # TODO: define a template
    output:
        os.path.join(preprocessing_pipeline_dir, template + '_PSD_raw.png')
        # TODO: define a template
    run:
        raw_path = Path(input[0]).with_suffix('.ds')
        raw = mne.io.read_raw_ctf(str(raw_path), preload=True, verbose=False)
        p = raw.plot_psd(show=False, fmax=150)
        p.axes[0].set_ylim(-30, 60)  # The limits were set so that all plots have the same vertical range
        p.savefig(output[0])
# TODO: name inputs and outputs


# Creates a pds plot of the data after linear filtering
rule draw_psd_linearly_filtered:
    input:
        os.path.join(preprocessing_pipeline_dir, template + '.fif')
        # TODO: define a template
    output:
        os.path.join(preprocessing_pipeline_dir, template + '_PSD_linearly_filtered.png')
        # TODO: use the template
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        p = raw.plot_psd(show=False, fmax=150)
        p.axes[0].set_ylim(-30, 60)
        p.savefig(output[0])
# TODO: name inputs and outputs


rule fit_ica:
    input:
        os.path.join(preprocessing_pipeline_dir, template + '.fif')
        # TODO: define a template
    output:
        os.path.join(preprocessing_pipeline_dir, template + '.ica')
        # TODO: define a template
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        # Remove the low-frequency part of the signal - standard step before ICA.
        raw.filter(1, None)
        # The number of components was selected without good reasons. Probably copied from a tutorial.
        ica = mne.preprocessing.ICA(random_state=2, n_components=25, verbose=False)
        ica.fit(raw)
        ica.save(output[0])
# TODO: name inputs and outputs


# Finds artifactual ICA components
rule find_ics:
    input:
        os.path.join(preprocessing_pipeline_dir, template + '.fif'),
        os.path.join(preprocessing_pipeline_dir, template + '.ica')
        # TODO: define templates
    output:
        os.path.join(preprocessing_pipeline_dir, template + '.ics.pickle'),
        os.path.join(preprocessing_pipeline_dir, template + '_ics_properties.pickle')
        # TODO: define templates
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(input[1])
        ics = find_ics_iteratively(raw, ica, verbose=False)
        pd.to_pickle(ics, output[0])
        # When ics is empty, plot all components (`ics or None` will evaluate to None)
        component_figures = ica.plot_properties(raw, picks=(ics or None), show=False)
        pd.to_pickle(component_figures, output[1])
# TODO: name inputs and outputs


rule remove_artifactual_ics:
    input:
        os.path.join(preprocessing_pipeline_dir, template + '.fif'),
        os.path.join(preprocessing_pipeline_dir, template + '.ica'),
        os.path.join(preprocessing_pipeline_dir, template + '.ics.pickle')
        # TODO: define templates
    output:
        os.path.join(preprocessing_pipeline_dir, template + '-ics-removed.fif')
        # TODO: define a template
    run:
        raw = mne.io.read_raw_fif(input[0], preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(input[1])
        ics = pd.read_pickle(input[2])
        ica.exclude = ics
        raw_ics_removed = ica.apply(raw)
        raw_ics_removed.save(output[0])
# TODO: name inputs and outputs


def inputs_for_report(wildcards):
    """
    Finds all the files necessary for the `make_preproc_report` rule to make a preprocessing report for a single
    subject. We had to put these in a function because the report file name does not have information about the session
    ids.
    :param wildcards: wildcards that triggered the rule, in this case just `subject`
    :return: a dict with all the inputs for the `make_preproc_report` rule
    """
    subject = wildcards.subject

    session1 = subjects_df.query('subject == @subject and session_number == 1').session_id.values[0]
    session2 = subjects_df.query('subject == @subject and session_number == 2').session_id.values[0]

    prefix1 = os.path.join(preprocessing_pipeline_dir, template.format(subject=subject, session=session1))
    prefix2 = os.path.join(preprocessing_pipeline_dir, template.format(subject=subject, session=session2))

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


# Runs a parameterized notebook in `01_preprocessing/report.ipynb` using the `papermill` package. Then converts it to
# HTML using `nbconvert`.
rule make_preproc_report:
    input:
        unpack(inputs_for_report)
    output:
        preprocessing_report_template + '_preproc_report.html'
        # TODO: define a template
    run:
        # create ipynb
        ipynb_path = Path(output[0]).with_suffix('.ipynb')
        _ = pm.execute_notebook(
            os.path.join('01_preprocessing', 'report.ipynb'),
            ipynb_path,
            parameters=input,
        )

        # convert to HTML
        html_exporter = HTMLExporter()
        html_exporter.template_name = 'classic'
        body, resources = html_exporter.from_file(ipynb_path)

        with Path(output[0]).open('wt') as f:
            f.write(body)

        # remove ipynb
        os.remove(str(ipynb_path))
# TODO: name inputs and outputs  


# This is a pseudo-rule in that it does not actually create any outputs - these are made manually after the inspection
# of the preprocessing report. So, if all the reports have been checked and the corresponding files with the report
# evaluations have been created then this rule won't be triggered at all because its outputs are already there and up to
# date. If, however, the evaluations don't exist or they are older than the corresponding report, this rule will be
# triggered and throw an error telling the user to create the manual file check.
rule manual_checks_done:
    input:
        report = rules.make_preproc_report.output[0]
    output:
        manual_check = manual_check_template
    run:
        raise ValueError('Error! Preprocessing report file does not exist!\n'
                          'Create the manualCheck file manually\n'
                          f'File: {output}\n')


# We don't know how to compare eigenspectra for different sets of sensors and thus will later use only the sensors that
# are present in all the recordings. This rule gets us this list.
rule common_channels:
    input:
        expand(linearly_filtered_template, zip, subject=subjects, session=sessions)
    output:
        common_channels_path
    run:
        common_channels = None
        for raw_path in input:
            raw = mne.io.read_raw_fif(raw_path, verbose=False)
            ch_names = set(raw.info['ch_names'])
            if common_channels is None:
                common_channels = ch_names
            else:
                # &= - set intersection
                common_channels &= ch_names

        pd.DataFrame(data=list(common_channels), columns=['common_channel']).to_csv(output[0], index=False)


# Calculates the correlation matrix, its normalized Laplacian and then the eigenvalues.
# One of the points of using the Laplacian is getting all the eigenvalues into the same 0-2 range where the spectra
# can be compared.
rule create_eigenvalues:
    input:
        cleaned_data = rules.remove_artifactual_ics.output[0],
        common_channels = common_channels_path
    output:
        eigenvalues = eigenvalues_template
    run:
        data = mne.io.read_raw_fif(input.cleaned_data, preload=True)
        # Channels that are present in all the recordings.
        common_channels = pd.read_csv(input.common_channels).common_channel.values.tolist()
        data = data.pick_channels(common_channels)
        # Removes miscs and the compensation channels.
        data_array = data.get_data(picks='mag')

        corr_mat = np.corrcoef(data_array)
        # We can't calculate a Laplacian if the sum in any of the rows is non-positive. One way to enforce this is to
        # substitute 0 for any negative values. We lose all the negative correlation this way though. There are other
        # options like taking only the negatives and flipping them, using the absolute values, etc. We should maybe
        # try them.
        corr_mat_pos = np.clip(corr_mat, 0, 1)
        laplac_mat = csgraph.laplacian(corr_mat_pos, normed=True)
        eigs = linalg.eigvals(laplac_mat)
        np.save(output.eigenvalues, eigs)
