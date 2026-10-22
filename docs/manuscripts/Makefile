PYTHON := python3
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error
BUILD := build

.PHONY: all paper book clean
all: paper book

paper:
	mkdir -p $(BUILD)/paper
	$(LATEXMK) -outdir=$(BUILD)/paper -jobname=bayesbreak-main-paper paper/main.tex

book:
	mkdir -p $(BUILD)
	$(LATEXMK) -outdir=$(BUILD) -jobname=bayesbreak-technical-book book/main.tex

clean:
	rm -rf $(BUILD)
