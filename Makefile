.PHONY: all 

all: model/*.py __main__.py
	pytest -s
	python .
