\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{scadsai-report-template}

\RequirePackage{kvoptions}
\SetupKeyvalOptions{%
    family=scadsai,%
    prefix=scadsai@%
}
% \RequirePackage{etoolbox}
% \usepackage{xifthen}
\RequirePackage{ifthen}

% Declare options
\DeclareStringOption{mainlanguage}

% Process options
\ProcessKeyvalOptions*
% Execute the default options
% \ProcessOptions\relax

% Access option values
% \newcommand{\mainlanguage}{\scadsai@mainlanguage} % replace with actual value
\newcommand*{\mainlanguage}{%
	\expandafter\scadsai@mainlanguage\expandafter{\CurrentOption}%
}

\newcommand*{\prilanguage}{english}
\newcommand*{\seclanguage}{ngerman}

\ifthenelse{\equal{\scadsai@mainlanguage}{english}}{%
	\renewcommand*{\prilanguage}{english}
	\renewcommand{\seclanguage}{ngerman}
}{}%
\ifthenelse{\equal{\scadsai@mainlanguage}{ngerman}}{%
	\renewcommand*{\prilanguage}{ngerman}
	\renewcommand*{\seclanguage}{english}
}{}%

% Now load the tudscrbook class
\LoadClass[
	paper = a4,
	BCOR=5mm, 			% BCOR je nach Seitenzahl setzen
	twoside,
	numbers=noenddot,
	parskip	= half, 	% separate paragraphs with half a line
	cdgeometry = symmetric,
	cd = barcolor,
    color = Black,
	chapterpage	= false,
	cdmath = false,
	slantedgreek=standard,
	captions=tableheading,
	subjectthesis=true,
	\seclanguage,\prilanguage % Give every language used in the document, the main one as last
]{tudscrbook}

\usepackage{tudscrsupervisor} % script for creatiing the task description
\usepackage{tudscrcolor}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}	

\usepackage[main=\prilanguage, \seclanguage]{babel}
% \usepackage{babel}
\usepackage{csquotes}

\usepackage[\prilanguage]{isodate} 
\usepackage{blindtext}

\usepackage{tikz}

\usepackage{setspace}
\usepackage{acronym}
\usepackage{scrhack} 	% acronyms result in warning without this
\usepackage{multicol} 	% use multiple columns, used for the acronyms section

\usepackage{enumitem}\setlist{noitemsep} % used for the bullet points in the task section
\usepackage{microtype}	% better spacing

\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{bm}			% used for making things bold in equations
\usepackage[b]{esvect}

\usepackage{multirow}	% use tables with columns stretching over multiple rows
\usepackage{booktabs}
\setlength\heavyrulewidth{0.25ex}
\usepackage{longtable}

\usepackage[font=footnotesize, format=plain, labelfont=bf]{caption}  % 
\usepackage{subcaption}	% Packages to allow subfigures

\usepackage[bottom, hang, flushmargin]{footmisc}
\renewcommand\hangfootparindent{1em}
%\usepackage{fnpct}

% use modern bib package
\usepackage[
	backend=biber,
	% style=alphabetic,
	% sorting=ynt
	]{biblatex}
\addbibresource{2_bib/references.bib}

\usepackage{xcolor}
\usepackage{listings}	% Package for displaying code
\definecolor{KeywordBlue}{cmyk}{0.88,0.77,0,0} %88,77,0,0
\definecolor{CommentGreen}{cmyk}{0.87,0.24,1.0,0.13} %87,24,100,13
\lstset{basicstyle=\scriptsize\ttfamily, language=C, commentstyle=\color{CommentGreen}, keywordstyle=\ttfamily\color{KeywordBlue}, backgroundcolor =\color[rgb]{0.95,0.95,0.95}, breaklines=true,literate={\\\%}{{\textcolor{black}{\\\%}}}1}

% \usepackage[a-1b]{pdfx}

% some of the metadata for the pdf are defined in the title-file,
% as there are variables like author and title, whích would appear twice otherwise
% hyperref should always be the last package to be loaded
\usepackage[
colorlinks=true,
urlcolor=.,
citecolor=.,
linkcolor=.,        
pdfstartview=FitV,                          		
pdfdisplaydoctitle=true,
hyperfootnotes=false
]{hyperref}
\urlstyle{same}		% use the same font for URLs as for the text

% %%% PARAMS %%%
% \pdfminorversion=7	% creates pdfs in the version 1.7, which prevents a warning with the logo

% Allow for triple digit page numbers in the toc
\makeatletter
\renewcommand*\@pnumwidth{2.1em}
\renewcommand*\@tocrmarg{3.1em}
\makeatother

\KOMAoptions{toc=chapterentrydotfill} 	% Add dots in toc for chapters
\setstretch{1.1}						% Adds a bit of space between the lines
\frenchspacing							% Only a single space after a dot


% Parameters to reduce 'Orphans' and 'Widdows'
\clubpenalty 			= 9999
\widowpenalty 			= 9999
\displaywidowpenalty   	= 1602
\brokenpenalty			= 4999	% Parameter for word disjuction on a pagebreak
\pretolerance			= 1100	% Parameter for difference from choosen format
\tolerance 				= 100 	% Parameter for difference from choosen format

% Less coservative parameters for floating objects in LaTeX
% An overview can be found in the book
% The Latex Companions Chapter 6.1
% A good start is
% http://robjhyndman.com/researchtips/latex-floats/

\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}
\renewcommand{\topfraction}{0.85}
\renewcommand{\bottomfraction}{0.85}
\renewcommand{\textfraction}{0.15}
\renewcommand{\floatpagefraction}{0.7}
\renewcommand{\textfraction}{0.1}
\setlength{\floatsep}{5pt plus 2pt minus 2pt}

% \colorlet{HKS41}{black}

%\setlength{\textfloatsep}{15pt plus 2pt minus 2pt}
%\setlength{\intextsep}{5pt plus 2pt minus 2pt}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basic information
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\ifTitle\Titlefalse % Set default to false
\newcommand{\getTitle}{\ifTitle\Titlevalue\else January 1, 2000\fi}
\newcommand{\Title}[1]{\Titletrue\def\Titlevalue{#1}}

\newif\ifAuthor\Authorfalse % Set default to false
\newcommand{\getAuthor}{\ifAuthor\Authorvalue\else Firstname Lastname\fi}
\newcommand{\Author}[1]{\Authortrue\def\Authorvalue{#1}}

\newif\ifDate\Datefalse % Set default to false
\newcommand{\getDate}{\ifDate\Datevalue\else January 1, 2000\fi}
\newcommand{\Date}[1]{\Datetrue\def\Datevalue{#1}}

\newif\ifDateOfBirth\DateOfBirthfalse % Set default to false
\newcommand{\getDateOfBirth}{\ifDateOfBirth\DateOfBirthvalue\else 30.02.2000\fi}
\newcommand{\DateOfBirth}[1]{\DateOfBirthtrue\def\DateOfBirthvalue{#1}}

\newif\ifPlaceOfBirth\PlaceOfBirthfalse % Set default to false
\newcommand{\getPlaceOfBirth}{\ifPlaceOfBirth\PlaceOfBirthvalue\else Dresden\fi}
\newcommand{\PlaceOfBirth}[1]{\PlaceOfBirthtrue\def\PlaceOfBirthvalue{#1}}

\newif\ifMatriculationNumber\MatriculationNumberfalse % Set default to false
\newcommand{\getMatriculationNumber}{\ifMatriculationNumber\MatriculationNumbervalue\else 123456789\fi}
\newcommand{\MatriculationNumber}[1]{\MatriculationNumbertrue\def\MatriculationNumbervalue{#1}}

\newif\ifReportType\ReportTypefalse % Set default to false
\newcommand{\getReportType}{\ifReportType\ReportTypevalue\else master\fi}
\newcommand{\ReportType}[1]{\ReportTypetrue\def\ReportTypevalue{#1}}

\newif\ifGraduation\Graduationfalse % Set default to false
\newcommand{\getGraduation}{\ifGraduation\Graduationvalue\else Master of Science\fi}
\newcommand{\getGraduationShort}{\ifGraduation\GraduationShortvalue\else M. Sc.\fi}
\newcommand{\Graduation}[2]{\Graduationtrue\def\Graduationvalue{#1}\def\GraduationShortvalue{#2}}

\newif\ifSupervisors\Supervisorsfalse % Set default to false
\newcommand{\getSupervisors}{\ifSupervisors\Supervisorsvalue\else master\fi}
\newcommand{\Supervisors}[1]{\Supervisorstrue\def\Supervisorsvalue{#1}}

\newif\ifReviewers\Reviewersfalse % Set default to false
\newcommand{\getReviewers}{\ifReviewers\Reviewersvalue\else master\fi}
\newcommand{\Reviewers}[1]{\Reviewerstrue\def\Reviewersvalue{#1}}

\newif\ifStartDate\StartDatefalse % Set default to false
\newcommand{\getStartDate}{\ifStartDate\StartDatevalue\else 01.01.1990\fi}
\newcommand{\StartDate}[1]{\StartDatetrue\def\StartDatevalue{#1}}

\newif\ifSubmissionDate\SubmissionDatefalse % Set default to false
\newcommand{\getSubmissionDate}{\ifSubmissionDate\SubmissionDatevalue\else 01.01.1990\fi}
\newcommand{\SubmissionDate}[1]{\SubmissionDatetrue\def\SubmissionDatevalue{#1}}

\newif\ifCourse\Coursefalse % Set default to false
\newcommand{\getCourse}{\ifCourse\Coursevalue\else abc\fi}
\newcommand{\Course}[1]{\Coursetrue\def\Coursevalue{#1}}

\newif\ifMatriculationYear\MatriculationYearfalse % Set default to false
\newcommand{\getMatriculationYear}{\ifMatriculationYear\MatriculationYearvalue\else abc\fi}
\newcommand{\MatriculationYear}[1]{\MatriculationYeartrue\def\MatriculationYearvalue{#1}}

\newcommand{\makefrontmatter}{
	\frontmatter
	\pagenumbering{Roman} 				% needed to capitalize the roman page numbering

	% Setting up Faculty, Insitute, Chair, etc information
  	\iflanguage{ngerman}
	{	
		\faculty{Fakultät Informatik}
		\institute{Institut für Technische Informatik}
		\chair{Professur für Rechnerarchitektur}
		\extraheadline{ScaDS.AI - Center for Scalable Data Analytics and Artificial Intelligence}
	}
	{	
		\faculty{Faculty of Computer Science}
		\institute{
			Institute of Computer Engineering%	
		} % \phantom{Push the institute name below the line.}}
		\chair{Chair of Computer Architecture}
		\extraheadline{ScaDS.AI - Center for Scalable Data Analytics and Artificial Intelligence}
		
	}
	
	\headlogo{logo/scadsai_cids_logo.png}
	
	% Setting information to the template
	\title{\getTitle}
	\author{\getAuthor}
	\dateofbirth{\getDateOfBirth}
	\placeofbirth{\getPlaceOfBirth}
	\matriculationnumber{\getMatriculationNumber}
	\subject{\getReportType}
	\thesis{\getReportType}
	\graduation[\getGraduationShort]{\getGraduation}
	\supervisor{\getSupervisors}
	\referee{\getReviewers}
	\startdate{\getStartDate}
	\issuedate{\getStartDate}
	\submissiondate{\getSubmissionDate}
	\duedate{\getSubmissionDate}
	\course{\getCourse}
	\matriculationyear{\getMatriculationYear}
	% Create title/cover page
	\maketitle
}	

\AtBeginDocument{
	\cleanlookdateon % removes the ordinal numbering of the day
}
\endinput
