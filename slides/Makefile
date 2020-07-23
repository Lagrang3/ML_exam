flags=-interaction=nonstopmode -halt-on-error
VPATH=metapost:tex:macros:images
PREFIX:=slides
IMAGES:=confusion.pdf error.pdf relu.pdf
PDFS:=$(PREFIX).pdf
MACS:=mac.tex
TEXS:=slides.tex

export

default:
	$(MAKE) $(MFLAGS) -C images
	$(MAKE) $(MFLAGS) $(PDFS)

$(IMAGES) : %.pdf : %.py
	$(MAKE) $(MFLAGS) -C images

all: slides.pdf 

$(PREFIX).pdf : $(PREFIX).tex $(MACS) $(IMAGES) $(TEXS)
	pdflatex $(flags) $<

clean:
	$(MAKE) $(MFLAGS) -C metapost clean
	-rm $(PREFIX).pdf *.log $(PREFIX).lp $(MPPDF) *.1

.DELETE_ON_ERROR: $(PREFIX).pdf

.PHONY: clean
