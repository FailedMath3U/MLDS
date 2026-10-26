LaTeX shuttle report
===================
Main source: SHUTTLE_REPORT.tex

Overleaf: upload this ZIP as a new project, choose SHUTTLE_REPORT.tex as the
main document, and compile with pdfLaTeX or XeLaTeX.

Local compilation, from the project folder:
  tectonic SHUTTLE_REPORT.tex
or:
  pdflatex SHUTTLE_REPORT.tex
  pdflatex SHUTTLE_REPORT.tex

Keep the included results/report_figures directory beside the .tex file.
The figures are vector PDFs generated from the saved solver results.
The report uses standard LaTeX packages and includes no external bibliography.

In the complete assignment folder, make_report_figures.py regenerates the
report figures from the saved NPZ and JSON files. The plotting notebook and
its original figures are also retained separately.

The earlier create_report_pdf.py is the superseded Python PDF renderer;
compile SHUTTLE_REPORT.tex to build the revised LaTeX report instead.
