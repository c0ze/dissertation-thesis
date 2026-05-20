INPUT = qucomp

all: $(INPUT).tex 
	latex	$(INPUT)
	bibtex	$(INPUT)
	bibtex	$(INPUT)
	latex	$(INPUT)
	latex	$(INPUT)
	dvipdf	$(INPUT)
	rm		$(INPUT).{aux,bbl,blg,log,toc}

viewpdf: $(INPUT).pdf
	acroread $(INPUT).pdf &

viewdvi: $(INPUT).pdf
	xdvi $(INPUT).dvi &


