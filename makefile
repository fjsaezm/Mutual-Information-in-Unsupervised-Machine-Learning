all: thesis clean

thesis: thesis.tex Chapters/1-Probability-Theory/* Chapters/2-Information-Theory/* Chapters/3-Representation-Learning/* Chapters/APPENDIX/*
	pdflatex --shell-escape thesis.tex
	bibtex thesis
	pdflatex --shell-escape thesis.tex
	pdflatex --shell-escape thesis.tex

clean:
	rm *.bbl *.aux *.blg *.fdb_latexmk *.out *.log *.toc *.fls
