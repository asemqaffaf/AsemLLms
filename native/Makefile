# Builds the C side of AsemLLM:
#   build/libtlm.so        kernels + TLM1 loader + inference; used from Lisp through sb-alien
#   build/tinyllm_nif.so   the same code as an Erlang NIF; `make install` copies it to the OTP app's priv/
# Options:  BLAS=cblas | accelerate   link a BLAS for big matrix products (faster, NOT bit-identical)
# -ffp-contract=off is required for bit-identical results with the Erlang and Lisp code (arm64 compilers
# fuse a*b+c into one rounding by default, the BEAM and SBCL do not).
CC     ?= cc
OPT    ?= -O3
CFLAGS += $(OPT) -std=c11 -fPIC -Wall -Wextra -Wno-unused-parameter -ffp-contract=off -fno-fast-math
LDLIBS += -lm
UNAME  := $(shell uname -s)

ifeq ($(UNAME),Darwin)
  SHARED := -dynamiclib -undefined dynamic_lookup
else
  SHARED := -shared
endif
ifeq ($(BLAS),cblas)
  CFLAGS += -DTLM_BLAS_CBLAS
  LDLIBS += -lopenblas
endif
ifeq ($(BLAS),accelerate)
  CFLAGS += -DTLM_BLAS_ACCELERATE
  LDLIBS += -framework Accelerate
endif

ERL_INCLUDE := $(shell erl -noshell -eval 'io:format("~s/erts-~s/include", [code:root_dir(), erlang:system_info(version)]), halt().' 2>/dev/null)
PRIV := ../tinyllm/apps/tinyllm/priv
SRC  := tlm_kernels.c tlm_model.c

all: build/libtlm.so
	@if [ -f "$(ERL_INCLUDE)/erl_nif.h" ]; then $(MAKE) --no-print-directory build/tinyllm_nif.so; \
	 else echo "erl_nif.h not found: skipping the Erlang NIF"; fi

build/libtlm.so: $(SRC) tlm.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(SHARED) -o $@ $(SRC) $(LDLIBS)

build/tinyllm_nif.so: $(SRC) tinyllm_nif.c tlm.h
	@mkdir -p build
	$(CC) $(CFLAGS) -I"$(ERL_INCLUDE)" $(SHARED) -o $@ $(SRC) tinyllm_nif.c $(LDLIBS)

install: all
	@if [ -f build/tinyllm_nif.so ]; then mkdir -p $(PRIV) && cp build/tinyllm_nif.so $(PRIV)/; fi

clean:
	rm -rf build $(PRIV)/tinyllm_nif.so

.PHONY: all install clean
