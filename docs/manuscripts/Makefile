PYTHON := python3
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error
BUILD := build

.PHONY: all paper book clean
all: paper book

paper:
	mkdir -p $(BUILD)
	cd bayesbreak-paper && $(LATEXMK) -outdir=../$(BUILD) -jobname=bayesbreak-paper main.tex

book:
	mkdir -p $(BUILD)
	cd bayesbreak-book && $(LATEXMK) -outdir=../$(BUILD) -jobname=bayesbreak-book book/main.tex

clean:
	rm -rf $(BUILD)
