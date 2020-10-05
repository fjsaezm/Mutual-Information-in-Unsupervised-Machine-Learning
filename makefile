all: eod

eod: eod.tex Chapters/probability.tex Chapters/info-theory.tex Chapters/appendix-a.tex
	pdflatex --shell-escape eod.tex
	bibtex eod
	pdflatex --shell-escape eod.tex
	pdflatex --shell-escape eod.tex
