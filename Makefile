TEX_MAIN=tex/main.tex
BUILD_DIR=build
XELATEX?=xelatex
BIBER?=biber
.PHONY: all clean

all: $(BUILD_DIR)/course.pdf

$(BUILD_DIR)/course.pdf: $(TEX_MAIN)
	mkdir -p $(BUILD_DIR)
	cd tex && $(XELATEX) -interaction=nonstopmode -halt-on-error -output-directory ../$(BUILD_DIR) main.tex
	cd $(BUILD_DIR) && $(BIBER) main
	cd tex && $(XELATEX) -interaction=nonstopmode -halt-on-error -output-directory ../$(BUILD_DIR) main.tex

clean:
	rm -rf $(BUILD_DIR)/*
