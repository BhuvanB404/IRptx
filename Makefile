# Makefile -- irptx core. C99, no dependencies beyond libc.
#
#   make          build/irptx, a normal (dynamic) build for development
#   make static   bin/irptx, linked -static; this is the binary that is committed
#   make test     bin/trunner (C unit tests; --list, --fam <fam>), tests/smoke.sh, Python unittests
#   make rules    bash tests/rules.sh
#   make clean    remove build/ and bin/trunner
#
# compile.py's fallback builds src/*.c with `cc -std=c99 -O2`; keep these
# sources and that language level in step with it.

CC     ?= cc
PY     ?= .venv/bin/python
CFLAGS  = -std=c99 -Wall -Wextra -pedantic -Werror -O2

SRCS  = $(sort $(wildcard src/*.c))
HDRS  = $(wildcard src/*.h)
TSRCS = $(sort $(wildcard tests/t_*.c))
TMAIN = $(filter-out src/main.c,$(SRCS))

all: build/irptx

build/irptx: $(SRCS) $(HDRS) Makefile
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $(SRCS)

# bin/irptx is committed (the grader needs no C compiler), so the static
# target writes straight into it. Rebuild and recommit whenever src/ changes.
static: bin/irptx

bin/irptx: $(SRCS) $(HDRS) Makefile
	@mkdir -p bin
	$(CC) $(CFLAGS) -static -o $@ $(SRCS)

# build/tcases.h lists every TH_REG test (tests/gencases.awk checks the names).
build/tcases.h: $(TSRCS) tests/gencases.awk
	@mkdir -p build
	awk -f tests/gencases.awk $(TSRCS) > $@.tmp && mv $@.tmp $@ || { rm -f $@.tmp; exit 1; }

bin/trunner: tests/tmain.c tests/tharns.h build/tcases.h $(TSRCS) $(SRCS) $(HDRS)
	@mkdir -p bin
	$(CC) $(CFLAGS) -Isrc -Itests -Ibuild -o $@ tests/tmain.c $(TSRCS) $(TMAIN)

test: bin/trunner
	./bin/trunner
	bash tests/smoke.sh
	$(PY) -m unittest tests.t_fe tests.t_cli
	$(PY) -m unittest discover -s tools -p 'test_*.py'

rules:
	bash tests/rules.sh

# bin/irptx is committed, so clean never removes it.
clean:
	rm -rf build bin/trunner

.PHONY: all static test rules clean
