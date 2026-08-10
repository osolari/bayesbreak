PYTHON := python3
LATEXMK := latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error
BUILD := build

.PHONY: all sync book paper paper-single paper-all executive \
	handoff-check presentation-check math-check validate validate-phase6 clean

all: validate

sync:
	$(PYTHON) scripts/render_canonical_handoff.py

book: sync
	mkdir -p $(BUILD)
	$(LATEXMK) -outdir=$(BUILD) -jobname=bayesbreak-technical-book book/main.tex

paper:
	mkdir -p $(BUILD)/paper
	$(LATEXMK) -outdir=$(BUILD)/paper -jobname=bayesbreak-main-paper paper/main-two-column.tex

paper-single:
	mkdir -p $(BUILD)/paper-single
	$(LATEXMK) -outdir=$(BUILD)/paper-single -jobname=bayesbreak-main-paper-single paper/main-single-column.tex

paper-all: paper paper-single

executive:
	mkdir -p $(BUILD)/executive
	$(LATEXMK) -outdir=$(BUILD)/executive -jobname=bayesbreak-executive-summary executive/main.tex

handoff-check: sync
	$(PYTHON) scripts/check_sync.py

presentation-check:
	$(PYTHON) scripts/check_presentation_handoffs.py

math-check:
	$(PYTHON) scripts/verify_phase6_math.py

validate-phase6: book paper-all executive handoff-check presentation-check math-check

validate: validate-phase6

clean:
	rm -rf $(BUILD)
