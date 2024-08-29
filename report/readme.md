# ScaDS.AI Report Template

This template can be used to write reports and thesis. Sample report can be found [here](../other/sample_report.pdf).

## How to use
This template can be used by following any methods mentioned [here](../readme.md).

## Parameters that can be used
To set the report language please set the preference at the top of [main.tex](main.tex) as shown below:

```latex
% For English
\documentclass[mainlanguage=english]{scadsai-report-template}

% For German
\documentclass[mainlanguage=ngerman]{scadsai-report-template}
```

Following report information needs to be mentioned:
```latex
% Title
\Title{My Report Title}
% Authour name: Firstname Lastname
\Author{Firstname Lastname}  
% Date of birth: DD.MM.YYYY
\DateOfBirth{17.02.2024} 
% Place of birth
\PlaceOfBirth{Dresden, Germany} 
% Matriculation number
\MatriculationNumber{123456789}
% Matriculation year: YYYY
\MatriculationYear{2024}
% Course
\Course{Computational Modelling and Simulation (CMS)}
% Report type: For more possible values, check readme.md of the report repository
\ReportType{master}
% Graduation: \Graduation{shortname}{longname}
\Graduation{M.Sc.}{Master of Science}
% Reviewers
\Reviewers{Prof. Dr. Wolfgang E. Nagel \and 2nd Referee}
% Supervisors: Seperated by '\and'
\Supervisors{Supervisor1 \and Supervisor2}
% Date of starting work: DD.MM.YYYY
\StartDate{02.02.2024}
% Date of submission: DD.MM.YYYY
\SubmissionDate{30.11.2024}
```

The possible values for `\ReportType` are as follows:
| Possible keywords | Bezeichner | Deutsch |  Englisch |
| -- | -- | -- | -- |
| habil | \habilitationname | Habilitation | Habilitation |
| diss | \dissertationname | Dissertation | Dissertation
| phd | \dissertationname|  Dissertation|  Dissertation
diploma | \diplomathesisname | Diplomarbeit | Diploma Thesis
master | \masterthesisname | Master-Arbeit | Master Thesis
bachelor | \bachelorthesisname | Bachelor-Arbeit | Bachelor Thesis
student | \studentthesisname | Studienarbeit | Student Thesis
evidence | \studentresearchname | Großer Beleg | Student Research Project
project | \projectpapername | Projektarbeit | Project Paper
seminar | \seminarpapername | Seminararbeit | Seminar Paper
term | \termpapername | Hausarbeit | Term Paper
research | \researchname | Forschungsbericht | Research Report
log | \logname | Protokoll | Log
report | \reportname | Bericht | Report
internship | \internshipname | Praktikumsbericht | Internship Report

## Disclaimer

This template is inspired by [this work](https://tu-dresden.de/ing/informatik/smt/cgv/studium/materialien?set_language=en).
