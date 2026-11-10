# native/: C library under the Erlang and Lisp code

    make -C native            # build/libtlm.so (Lisp, via sb-alien) and build/tinyllm_nif.so (Erlang)
    make -C native install    # also copies the NIF to tinyllm/apps/tinyllm/priv/
    make -C native BLAS=cblas # optional: BLAS for big products (faster, NOT bit-identical; BLAS=accelerate on macOS)
    sh native/interop_test.sh # Lisp writes a model, the Erlang NIF must reproduce Lisp's logits exactly

What is in it
- `tlm_kernels.c`: the three matrix products of `lisp/src/tf.lisp` (`tlm_mm_nn/nt/tn`).
- `tlm_model.c`: TLM1 loader (mmap) and a one-token transformer step with a KV cache, for both norm kinds. See `FORMAT.md`.
- `tinyllm_nif.c`: model and cache as NIF resources. Calls: `load`, `cache_new`, `step`, `last_logits` (+ `_dirty` variants).

Design rules
- **Same numbers.** Every sum is added in the order the Erlang and Lisp code add it, and the Makefile passes `-ffp-contract=off`
  (arm64 compilers otherwise fuse multiply-add, which changes the last bit). The tests compare with equality, not tolerance.
- **The Erlang API does not change.** `tinyllm_tf:step/advance/last_logits/cache_truncate` dispatch through the
  `tinyllm_backend` behaviour: `tinyllm_backend_erl` (the original code) or `tinyllm_backend_nif`, chosen by the data
  (`params => #{nif := Res}`, cache `#{res := Res}`). `tinyllm_session` is untouched.
  `{ok, Native} = tinyllm_backend_nif:to_native(Model, "model.tlm")` gives a model every existing function accepts.
- **Cache = (buffer, N), copy on write.** The buffer is extended in place only when N is the number of positions already written;
  otherwise the first N positions are copied. A session that truncates and continues copies at most once per truncate.
- **Lisp falls back.** `native-init` runs at load; without the library the Lisp kernels run as before. Threshold for the C call:
  `TLM_MIN_WORK` (default 4096 = m*k*n); `bin/bench-native.lisp` shows where it pays on your machine.

Not done, on purpose or not yet
- Inference only: training (`loss_grad`, the Lisp backward pass beyond the three products) still runs in Erlang / Lisp.
- No GPU code. `BLAS=accelerate` uses the CPU (and AMX on Apple silicon) and was not compiled on a Mac here; `BLAS=cblas` was built and run on Linux.
- No ALiBi / relative positions: the models use learned absolute positions, so a full window is still rebuilt (see `tinyllm_session`).
  The cache buffer is not a ring for that reason.
- f64 only.
