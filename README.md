# BIOMAG2022 competition "Ketamine in Depression"

This repository contains the code for our stab at the [BIOMAG2022](https://www.biomag2020.org/) data analysis
competition ["Ketamine in Depression"](https://www.biomag2020.org/awards/data-analysis-competitions/).
The dataset includes MEG resting-state DATA from 35 participants (22 with MDD, 14 healthy).
Each participant had two sessions: one under ketamine and one sober (after taking placebo).
The recording are unlabelled, we only know which sessions belong to the same participant.
Our task is to either classify subjects as depressed/healthy or to classify sessions as ketamine/placebo.

Here is the description from the
[BIOMAG 2020 Data Analysis Competions page](https://www.biomag2020.org/awards/data-analysis-competitions/) (as of Apr, 
23 2021):

> **Organisers**: Section on the Neurobiology and Treatment of Mood Disorders, National Institute of Mental Health, National Institutes of Health, Bethesda MD USA
> 
> **Contact**: Jessica Gilbert, PhD, jessica.gilbert@nih.gov
> 
> **Data**: Resting state MEG data from a clinical trial of ketamine’s mechanisms of action in major depression
> 
> The data include 250 second eyes-closed resting-state data from a randomized, double-blind, placebo-controlled study of ketamine’s mechanisms of action. Data for the competition come from a total of 36 male and female participants aged 18–65 years old, 22 of which had major depression (MDD) and 14 of which were healthy control subjects. Subjects with MDD had been diagnosed with recurrent MDD without psychotic features using the Structured Clinical Interview for Axis I DSM-IV Disorders (SCID)-Patient Version. Subjects with MDD were required to have a score greater than 20 on the Montgomery-Åsberg Depression Rating Scale (MADRS) at screening. MDD participants were considered treatment-resistant and had to have not responded to at least one adequate antidepressant trial during their current episode, as assessed using the Antidepressant Treatment History Form and the current episode had to have lasted at least four weeks. Subjects were free from psychotropic medications in the two weeks before randomization (five weeks for fluoxetine, three weeks for aripiprazole). Healthy control subjects had no Axis I disorder as determined by SCID-NP, and no family history of Axis I disorders in first degree relatives. Healthy control subjects were also free of medications affecting neuronal function or cerebral blood flow or metabolism. Subjects in both groups were in good physical health as determined by medical history, physical exam, blood labs, electrocardiogram, chest x-ray, urinalysis, and toxicology. 
> 
> The competition data include two resting-state scans per participant, one occurring 6-to-9 hours following ketamine administration and one occurring 6-to-9 hours following placebo saline administration. These data were collected using a CTF Omega 275-channel system. The data have been de-identified, so each participant has been given a randomized 8-letter code at the beginning of the filename. You will also find the date of the scan in the filename (i.e., YYYYMMDD). Each scan occurred approximately 14-days apart, and participants were randomized to receive either ketamine or placebo-saline during their first infusion. You can approach the data analysis competition in one of two ways. First, you can attempt to classify participants with MDD from healthy control subjects (i.e., a between-subjects factor). Second, you can attempt to classify the scan session (i.e., a within-subjects factor: ketamine versus placebo). In either case, you should submit a short, written report detailing how you approached the data analysis (i.e., describing whether you classified based on the between-subjects or within-subjects factor), the methods used, and defining which scan(s) correspond to your chosen grouping of interest. You will receive one point for each scan that is correctly classified at either the within or between-subjects level. 
> 
> **Deadline**: Monday 2nd May 2022
> 
> **Limitations**: You are not allowed to use/distribute the data for any purposes irrelevant to the competition, without the organizer’s permission.

Team (in the order of the increasing first name length):
- Egor Levchenko
- Kirill Stepanovskikh
- Evgenii Kalenkovich
