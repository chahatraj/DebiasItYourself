MAIN = main

.PHONY: pdf force info text

pdf:
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

force:
	latexmk -g -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex

info:
	pdfinfo $(MAIN).pdf

text:
	pdftotext $(MAIN).pdf -
