all: thesis

eod: thesis.tex Chapters/probability.tex Chapters/info-theory.tex Chapters/appendix-a.tex
	pdflatex --shell-escape thesis.tex
	bibtex thesis
	pdflatex --shell-escape thesis.tex
	pdflatex --shell-escape thesis.tex
