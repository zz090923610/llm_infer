#include "generate.h"
#include <time.h>

static double monotonic_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int is_stop(Tokenizer *tok, int tid) {
    for (int i = 0; i < tok->n_stop; i++)
        if (tok->stop_ids[i] == tid) return 1;
    return 0;
}

void generate_state_init(GenerateState *st, LlamaModel *model, Tokenizer *tok, KVCache *cache,
                         int max_tokens, float temperature, int top_k, float top_p) {
    memset(st, 0, sizeof(*st));
    st->model = model;
    st->tok = tok;
    st->own_cache = cache == NULL;
    st->cache = cache ? cache : llama_model_new_cache(model, 1, model->hparams.n_ctx);
    st->max_tokens = max_tokens;
    st->temperature = temperature;
    st->top_k = top_k;
    st->top_p = top_p;
    rng_seed(&st->rng, (uint64_t)time(NULL) ^ 0xA5A5A5A5ULL);
}

void generate_state_free(GenerateState *st) {
    if (st->own_cache) kvcache_free(st->cache);
}

int generate_start(GenerateState *st, const int *prompt_ids, int n_prompt) {
    st->logits = llama_model_forward(st->model, prompt_ids, 1, n_prompt, st->cache, NULL, NULL);
    st->n_generated = 0;
    st->done = 0;
    return 0;
}

int generate_next(GenerateState *st, int *token_out) {
    if (st->done || st->n_generated >= st->max_tokens) {
        st->done = 1;
        return 1;
    }
    int tid = sample_token(st->logits, st->model->hparams.n_vocab, st->temperature, st->top_k, st->top_p,
                           &st->rng);
    if (is_stop(st->tok, tid)) {
        st->done = 1;
        return 1;
    }
    *token_out = tid;
    st->n_generated++;
    st->logits = llama_model_forward(st->model, &tid, 1, 1, st->cache, NULL, NULL);
    return 0;
}

char *generate_text(LlamaModel *model, Tokenizer *tok, const char *prompt, int max_tokens,
                    float temperature, int top_k, float top_p, int parse_special, int stream,
                    KVCache *cache) {
    IntVec ids;
    intvec_init(&ids);
    tokenizer_encode(tok, prompt, parse_special, &ids);
    GenerateState st;
    generate_state_init(&st, model, tok, cache, max_tokens, temperature, top_k, top_p);
    StreamDecoder dec;
    stream_decoder_init(&dec, tok, 1);
    ByteVec pieces;
    bytevec_init(&pieces);
    double t0 = monotonic_now();
    generate_start(&st, ids.data, ids.n);
    int n = 0;
    int tid;
    while (generate_next(&st, &tid) == 0) {
        n++;
        char *chunk = stream_decoder_push(&dec, tid);
        bytevec_append(&pieces, chunk, (int)strlen(chunk));
        if (stream && chunk[0]) {
            fputs(chunk, stdout);
            fflush(stdout);
        }
        free(chunk);
    }
    char *tail = stream_decoder_flush(&dec);
    bytevec_append(&pieces, tail, (int)strlen(tail));
    if (stream && tail[0]) {
        fputs(tail, stdout);
        fflush(stdout);
    }
    free(tail);
    if (stream) {
        double elapsed = monotonic_now() - t0;
        double tps = elapsed > 0.0 ? (double)n / elapsed : 0.0;
        printf("\n[%d tokens, %.2f tok/s]\n", n, tps);
        fflush(stdout);
    }
    bytevec_push(&pieces, 0);
    stream_decoder_free(&dec);
    generate_state_free(&st);
    intvec_free(&ids);
    return pieces.data;
}

static void as_batch(int **id_lists, const int *lengths, int n, int pad_id, int slen, int *batch) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < slen; j++) batch[i * slen + j] = pad_id;
        for (int j = 0; j < lengths[i]; j++) batch[i * slen + j] = id_lists[i][j];
    }
}

IntVec *generate_batch(LlamaModel *model, Tokenizer *tok, char **prompts, int n_prompts,
                       int max_tokens, float temperature, int top_k, float top_p, int parse_special) {
    LlamaHParams *hp = &model->hparams;
    IntVec *encoded = xcalloc((size_t)n_prompts, sizeof(IntVec));
    int *lengths = xmalloc((size_t)n_prompts * sizeof(int));
    int slen = 0;
    for (int i = 0; i < n_prompts; i++) {
        intvec_init(&encoded[i]);
        tokenizer_encode(tok, prompts[i], parse_special, &encoded[i]);
        lengths[i] = encoded[i].n;
        if (encoded[i].n > slen) slen = encoded[i].n;
    }
    int **id_lists = xmalloc((size_t)n_prompts * sizeof(int *));
    for (int i = 0; i < n_prompts; i++) id_lists[i] = encoded[i].data;
    int *batch = xmalloc((size_t)n_prompts * (size_t)slen * sizeof(int));
    as_batch(id_lists, lengths, n_prompts, hp->pad_id, slen, batch);

    KVCache *cache = llama_model_new_cache(model, n_prompts, hp->n_ctx);
    float *logits = llama_model_forward(model, batch, n_prompts, slen, cache, lengths, NULL);

    IntVec *out = xcalloc((size_t)n_prompts, sizeof(IntVec));
    for (int i = 0; i < n_prompts; i++) intvec_init(&out[i]);
    int *done = xcalloc((size_t)n_prompts, sizeof(int));
    Rng rng;
    rng_seed(&rng, 1);
    int *nxt = xmalloc((size_t)n_prompts * sizeof(int));
    int *valid = xmalloc((size_t)n_prompts * sizeof(int));

    for (int step = 0; step < max_tokens; step++) {
        int all_done = 1;
        for (int b = 0; b < n_prompts; b++) {
            if (done[b]) {
                nxt[b] = hp->eos_id;
                continue;
            }
            all_done = 0;
            int tid = sample_token(logits + (size_t)b * (size_t)hp->n_vocab, hp->n_vocab, temperature,
                                   top_k, top_p, &rng);
            if (is_stop(tok, tid)) {
                nxt[b] = hp->eos_id;
                done[b] = 1;
                continue;
            }
            nxt[b] = tid;
            intvec_push(&out[b], tid);
            if (tid == hp->eos_id) done[b] = 1;
        }
        if (all_done) break;
        for (int b = 0; b < n_prompts; b++) valid[b] = done[b] ? 0 : 1;
        logits = llama_model_forward(model, nxt, n_prompts, 1, cache, valid, NULL);
    }

    kvcache_free(cache);
    for (int i = 0; i < n_prompts; i++) intvec_free(&encoded[i]);
    free(encoded);
    free(lengths);
    free(id_lists);
    free(batch);
    free(done);
    free(nxt);
    free(valid);
    return out;
}
