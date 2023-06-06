% LaTex class file for generating KGCOE reports
% https://github.com/DeepHorizons/KGCOEReport_template

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{KGCOEReport}[2016/05/25 Example LaTeX class]

%%%%%%% Base class
\LoadClass[12pt]{article}

%%%%%%% Required packages
\RequirePackage{multicol} %Make columns
\RequirePackage[margin=1in]{geometry} %Change margins
\RequirePackage[most]{tcolorbox}  % For the "RIT" box on EEEE cover

\setlength{\parindent}{0pt} % Don't indent paragraphs

%%%%%%% Internal functions and environments
% Used to make sure the TA listing is even
\newenvironment{_TAalignment}{%
\def\\{\newline\leavevmode\phantom{TA:}}
}

%%%%%%% External functions and environments

%This pretty prints an equation.
%1st arg: The label to be used for reference
%2nd arg: The equation. Use Math Mode.
%3rd arg: The caption text.
\newcommand{\eq}[3]{
	\begin{equationCaption}[H]
				\begin{equation}
						#2
						\label{#1}
				\end{equation}
				\caption{#3}
	\end{equationCaption}
}

%%%%%%% Write a CE report
\DeclareOption{CMPE}{
	\newcommand{\KGCOEDocType}{CMPE}

	\renewcommand{\maketitle}{%
	\vspace*{2.5cm}

		\begin{center}
			\large
			\textbf{\classCode \ Homework \exerciseNumber}

			\bigskip

			\normalsize
		\end{center}

		\vspace*{5cm}

		\begin{multicols}{2}
			\phantom{LaTeX doesn't like empty columns} % Phantom will take up that much space, but not actually appear
			\columnbreak{} % Starts the second column
			\begin{flushleft} % Make sure LaTeX does not try to justify these lines
				\name\\
				TA:\begin{_TAalignment}\TAs  % There can't be a space between the \TAs and the enviroment declaration
				\end{_TAalignment}
				Lecture Section: \LectureSectionNum\\
				Professor: \LectureInstructor\\
				Submission Date: \submissionDate\\
			\end{flushleft}
		\end{multicols}


		\vspace{2cm}

		By submitting this report, you attest that you neither have given nor have received any assistance (including writing, collecting data, plotting figures, tables or graphs, or using previous student reports as a reference), and you further acknowledge that giving or receiving such assistance will result in a failing grade for this course.

		\vfill

		{\centering{}
		Your Signature: \hrulefill}

		\newpage
	}
}
%%%%%%% Write a EE report
\DeclareOption{EEEE}{
	\newcommand{\KGCOEDocType}{EEEE}

	\renewcommand{\maketitle}{%

		\begin{flushright}\begin{tcolorbox}[colback=black,coltext=white,halign=center,boxrule=0pt,
			sharp corners, width=0.5\textwidth, fontupper={\Large \bfseries}]
			Rochester Institute of \\ Technology
		\end{tcolorbox}\end{flushright}

		\huge
		\textbf{\classCode \ - \className \ Technical Memo}\\
		\normalsize

		\noindent
		\begin{tabular}{ll}
		\textbf{From:} &\name\\

		% Optional parameter of partner
		\ifdefined\partnerName
			\textbf{Partner:} &\partnerName\\
			\\
		\fi

		\textbf{To:} &\TAs \ | Section \LabSectionNum\\
		\textbf{Date:} &Performed: \dateDone; Due: \dateSubmitted\\
		\textbf{Subject:} &Lab \exerciseNumber: \exerciseDescription
		\end{tabular}

		\noindent
		\rule{\textwidth}{.1pt}
	}
}
% Pass any other options to
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions

%%%%%%% Check to see if we were given a correct department code
\@ifundefined{KGCOEDocType}{\ClassError{KGCOEReport}{Unrecognized Department Code}{Available department codes are CMPE, EEEE}}{}
