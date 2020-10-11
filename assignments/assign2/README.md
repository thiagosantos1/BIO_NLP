ASSIGNMENT 2

Version 1.0

1 DOMAIN-SPECIFIC PROBLEM DESCRIPTION

Participants

Participants in this study were adults previously diagnosed with Parkinson’s disease by a movement disorders specialist. All participants provided written informed consent prior to participation according to procedures approved by the Institutional Review Board of Emory University.

Study design

One-year prospective observational study. All participants were assessed with a detailed cognitive and motor battery of indicators of potential fall risk during a single study visit. Details of the study methodology are available in previous reports.1,2 After enrollment, study participants were prospectively tracked for incident falls for a nominal one year period. During the observation period, they were queried about the presence and circumstances of any falls at monthly intervals using mail, email, phone, or text, at the discretion of the participant. Details of subsequent falls were recorded by participants and verified by study staff. Approximately 1/3 or participants enrolled went on to fall during the observation period.

Fall reports

Patients were instructed to describe the date and location of each fall. They were asked to describe what they were doing when the fall occurred, what they thought caused the fall, and how they recovered afterwards. Falls were defined as “an unexpected event in which the participants come to rest on the ground, floor, or lower level.”3 In some cases the text have been edited or summarized by study staff.

Data description

The data contain descriptions of 116 unique falls reported by 24 unique participants. The time since enrollment in days are included. Each fall has been annotated as “CoM,” “BoS,” or “Other” by an expert rater.

Study question

To what extent can features extracted from free text (and potentially clinical variables) discriminate “CoM” from other fall types?

2 TASKS

Overview

Your task is to model this problem as a binary classification task: CoM vs. Other. Then you will have to design and execute a thorough NLP-driven study to:

(i)implement an automatic classifier
(ii)identify text features that are indicative of the classes
Minimum set of specific tasks

Implement an automatic classifier
Cross-validate on training set
Tune hyperparameters
Apply some sort of ensemble classification
Compare at least 5 classifiers + a Naive Bayes baseline
Engineer at least 4 features + n-grams
Identify the best classifier & feature set combination
Evaluate performance of classifiers based on micro-averaged F1 score for the CoM class. However, report all of the following in all evaluations:
Accuracy, micro-averaged F1 score, and macro-averaged F1 score.
For the best classifier:
Perform training set size vs. performance graph. Is the learning improving with data? Can we estimate how much annotated data we need?
Perform an ablation study (re-run experiments with one feature set removed at a time)