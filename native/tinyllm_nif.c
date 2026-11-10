/* Erlang NIF over libtlm: an mmap'd TLM1 model and KV caches as opaque resources.
 *
 * tinyllm_nif:load(Path)                       -> {ok, Model, Info} | {error, Reason}
 * tinyllm_nif:cache_new(Model)                 -> Cache
 * tinyllm_nif:step(Model, Cache, N, Id, Mode)  -> {Cache1, Logits | undefined}      Mode = full | kv_only
 * tinyllm_nif:last_logits(Model, Ids)          -> Logits
 * (step_dirty/5 and last_logits_dirty/2 are the same functions scheduled on a dirty CPU scheduler.)
 *
 * A cache value in Erlang is (resource, N): the first N positions of the resource's buffer. Erlang terms are
 * immutable, so the buffer is only extended in place when N equals the number of positions already written
 * ("committed"); otherwise another value still sees a longer prefix of the same buffer and the first N
 * positions are copied into a fresh buffer first (copy on write). A linear session (truncate, advance,
 * advance, ...) therefore costs at most one copy per truncate and none per token.
 */
#include <erl_nif.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "tlm.h"

typedef struct { tlm_model *m; } model_res;
typedef struct { tlm_cache *c; ErlNifMutex *mu; model_res *owner; } cache_res;

static ErlNifResourceType *MODEL_T, *CACHE_T;

static void model_dtor(ErlNifEnv *env, void *obj) { (void)env; tlm_close(((model_res *)obj)->m); }
static void cache_dtor(ErlNifEnv *env, void *obj) {
    (void)env;
    cache_res *r = (cache_res *)obj;
    tlm_cache_free(r->c);
    if (r->mu) enif_mutex_destroy(r->mu);
    if (r->owner) enif_release_resource(r->owner);
}

static ERL_NIF_TERM atom(ErlNifEnv *env, const char *s) {
    ERL_NIF_TERM a;
    return enif_make_existing_atom(env, s, &a, ERL_NIF_LATIN1) ? a : enif_make_atom(env, s);
}

static ERL_NIF_TERM err_term(ErlNifEnv *env, const char *msg) {
    return enif_make_tuple2(env, atom(env, "error"), enif_make_string(env, msg, ERL_NIF_LATIN1));
}

static ERL_NIF_TERM make_cache(ErlNifEnv *env, model_res *owner, tlm_cache *c) {
    cache_res *r = (cache_res *)enif_alloc_resource(CACHE_T, sizeof(cache_res));
    if (!r) { tlm_cache_free(c); return enif_make_badarg(env); }
    r->c = c;
    r->mu = enif_mutex_create("tlm_cache");
    r->owner = owner;
    enif_keep_resource(owner);
    ERL_NIF_TERM t = enif_make_resource(env, r);
    enif_release_resource(r);
    return t;
}

static void put(ErlNifEnv *env, ERL_NIF_TERM *map, const char *k, ERL_NIF_TERM v) {
    enif_make_map_put(env, *map, atom(env, k), v, map);
}

static ERL_NIF_TERM nif_load(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    ErlNifBinary bin;
    if (!enif_inspect_iolist_as_binary(env, argv[0], &bin) || bin.size == 0 || bin.size > 4095) return enif_make_badarg(env);
    char path[4096];
    memcpy(path, bin.data, bin.size);
    path[bin.size] = 0;
    tlm_model *m;
    char err[200];
    int rc = tlm_open(path, &m, err, sizeof err);
    if (rc != TLM_OK) return err_term(env, err);
    model_res *r = (model_res *)enif_alloc_resource(MODEL_T, sizeof(model_res));
    if (!r) { tlm_close(m); return err_term(env, "out of memory"); }
    r->m = m;
    ERL_NIF_TERM res = enif_make_resource(env, r);
    enif_release_resource(r);
    ERL_NIF_TERM info = enif_make_new_map(env);
    put(env, &info, "vocab", enif_make_uint(env, m->vocab));
    put(env, &info, "context", enif_make_uint(env, m->ctx));
    put(env, &info, "dim", enif_make_uint(env, m->dim));
    put(env, &info, "heads", enif_make_uint(env, m->heads));
    put(env, &info, "layers", enif_make_uint(env, m->layers));
    put(env, &info, "ff", enif_make_uint(env, m->ff));
    put(env, &info, "norm", atom(env, m->norm == TLM_NORM_LAYER ? "layer" : "rms"));
    put(env, &info, "biases", atom(env, (m->flags & TLM_FLAG_BIASES) ? "true" : "false"));
    put(env, &info, "params", enif_make_uint64(env, tlm_param_count(m)));
    put(env, &info, "bytes", enif_make_uint64(env, m->map_len));
    return enif_make_tuple3(env, atom(env, "ok"), res, info);
}

static ERL_NIF_TERM nif_cache_new(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    model_res *mr;
    if (!enif_get_resource(env, argv[0], MODEL_T, (void **)&mr)) return enif_make_badarg(env);
    tlm_cache *c = tlm_cache_new(mr->m);
    if (!c) return enif_raise_exception(env, atom(env, "enomem"));
    return make_cache(env, mr, c);
}

static ERL_NIF_TERM logits_term(ErlNifEnv *env, const double *lg, size_t n) {
    ERL_NIF_TERM *arr = (ERL_NIF_TERM *)enif_alloc(sizeof(ERL_NIF_TERM) * n);
    if (!arr) return enif_raise_exception(env, atom(env, "enomem"));
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(lg[i])) { enif_free(arr); return enif_raise_exception(env, atom(env, "nonfinite_logits")); }
        arr[i] = enif_make_double(env, lg[i]);
    }
    ERL_NIF_TERM l = enif_make_list_from_array(env, arr, (unsigned)n);
    enif_free(arr);
    return l;
}

static ERL_NIF_TERM rc_exception(ErlNifEnv *env, int rc) {
    switch (rc) {
    case TLM_E_CTX:   return enif_raise_exception(env, atom(env, "context_full"));
    case TLM_E_TOKEN: return enif_raise_exception(env, atom(env, "bad_token"));
    case TLM_E_MEM:   return enif_raise_exception(env, atom(env, "enomem"));
    default:          return enif_make_badarg(env);
    }
}

static ERL_NIF_TERM nif_step(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    model_res *mr; cache_res *cr; unsigned n, id; char mode[16];
    if (!enif_get_resource(env, argv[0], MODEL_T, (void **)&mr) ||
        !enif_get_resource(env, argv[1], CACHE_T, (void **)&cr) ||
        !enif_get_uint(env, argv[2], &n) || !enif_get_uint(env, argv[3], &id) ||
        !enif_get_atom(env, argv[4], mode, sizeof mode, ERL_NIF_LATIN1) || cr->owner != mr)
        return enif_make_badarg(env);
    int kv_only = strcmp(mode, "kv_only") == 0;
    if (!kv_only && strcmp(mode, "full") != 0) return enif_make_badarg(env);

    const tlm_model *m = mr->m;
    double *logits = NULL;
    if (!kv_only && !(logits = (double *)enif_alloc(sizeof(double) * m->vocab)))
        return enif_raise_exception(env, atom(env, "enomem"));

    ERL_NIF_TERM cache_term = argv[1];
    enif_mutex_lock(cr->mu);
    if (n > cr->c->committed) { enif_mutex_unlock(cr->mu); enif_free(logits); return enif_make_badarg(env); }
    int rc;
    if (cr->c->committed == n) {
        rc = tlm_run_token(m, cr->c, n, id, kv_only, logits);
        enif_mutex_unlock(cr->mu);
    } else {
        tlm_cache *copy = tlm_cache_clone(cr->c, n);
        enif_mutex_unlock(cr->mu);
        if (!copy) { enif_free(logits); return enif_raise_exception(env, atom(env, "enomem")); }
        rc = tlm_run_token(m, copy, n, id, kv_only, logits);
        if (rc != TLM_OK) { tlm_cache_free(copy); enif_free(logits); return rc_exception(env, rc); }
        cache_term = make_cache(env, mr, copy);
    }
    if (rc != TLM_OK) { enif_free(logits); return rc_exception(env, rc); }
    ERL_NIF_TERM lt = kv_only ? atom(env, "undefined") : logits_term(env, logits, m->vocab);
    enif_free(logits);
    return enif_make_tuple2(env, cache_term, lt);
}

static ERL_NIF_TERM nif_last_logits(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc;
    model_res *mr;
    if (!enif_get_resource(env, argv[0], MODEL_T, (void **)&mr)) return enif_make_badarg(env);
    const tlm_model *m = mr->m;
    unsigned len;
    if (!enif_get_list_length(env, argv[1], &len) || len == 0) return enif_make_badarg(env);
    if (len > m->ctx) return rc_exception(env, TLM_E_CTX);
    int32_t *ids = (int32_t *)enif_alloc(sizeof(int32_t) * len);
    double *logits = (double *)enif_alloc(sizeof(double) * m->vocab);
    if (!ids || !logits) { enif_free(ids); enif_free(logits); return enif_raise_exception(env, atom(env, "enomem")); }
    ERL_NIF_TERM l = argv[1], h;
    for (unsigned i = 0; i < len; i++) {
        int v;
        if (!enif_get_list_cell(env, l, &h, &l) || !enif_get_int(env, h, &v) || v < 0) {
            enif_free(ids); enif_free(logits); return enif_make_badarg(env);
        }
        ids[i] = v;
    }
    int rc = tlm_last_logits(m, ids, len, logits);
    enif_free(ids);
    if (rc != TLM_OK) { enif_free(logits); return rc_exception(env, rc); }
    ERL_NIF_TERM lt = logits_term(env, logits, m->vocab);
    enif_free(logits);
    return lt;
}

static ERL_NIF_TERM nif_abi(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    (void)argc; (void)argv;
    return enif_make_tuple2(env, enif_make_uint(env, TLM_VERSION), enif_make_int(env, tlm_blas_enabled()));
}

static int on_load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
    (void)priv; (void)info;
    MODEL_T = enif_open_resource_type(env, NULL, "tlm_model", model_dtor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    CACHE_T = enif_open_resource_type(env, NULL, "tlm_cache", cache_dtor, ERL_NIF_RT_CREATE | ERL_NIF_RT_TAKEOVER, NULL);
    return (MODEL_T && CACHE_T) ? 0 : 1;
}

static ErlNifFunc funcs[] = {
    {"abi", 0, nif_abi, 0},
    {"load", 1, nif_load, 0},
    {"cache_new", 1, nif_cache_new, 0},
    {"step", 5, nif_step, 0},
    {"step_dirty", 5, nif_step, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"last_logits", 2, nif_last_logits, 0},
    {"last_logits_dirty", 2, nif_last_logits, ERL_NIF_DIRTY_JOB_CPU_BOUND},
};

ERL_NIF_INIT(tinyllm_nif, funcs, on_load, NULL, NULL, NULL)
