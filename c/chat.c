#define _POSIX_C_SOURCE 200809L
#include "generate.h"
#include "gguf.h"
#include "backend.h"
#include "util.h"
#include <time.h>

static double monotonic_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#ifndef LLM_DEFAULT_MODEL
#define LLM_DEFAULT_MODEL "models/smollm2-360m-instruct-q8_0.gguf"
#endif

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--model PATH] [--max-tokens N] [--temp F] [--top-p F] [--top-k N] [--ctx N] "
            "[--think] [--threads N] [--prefill-threads N] [--decode-threads N] [--tok-tables PATH] "
            "[--quiet] [--steps]\n",
            argv0);
}

static char *strip_end_marker(char *reply) {
    size_t n = strlen(reply);
    while (n && (reply[n - 1] == ' ' || reply[n - 1] == '\n' || reply[n - 1] == '\t')) reply[--n] = 0;
    if (n >= 4 && strcmp(reply + n - 4, " END") == 0) {
        reply[n - 4] = 0;
        n -= 4;
    } else if (n >= 3 && strcmp(reply + n - 3, "END") == 0) {
        reply[n - 3] = 0;
        n -= 3;
    }
    while (n && (reply[n - 1] == ' ' || reply[n - 1] == '\n' || reply[n - 1] == '\t')) reply[--n] = 0;
    return reply;
}

static int prefix_equal(const int *a, int na, const int *b, int nb) {
    if (na > nb) return 0;
    for (int i = 0; i < na; i++)
        if (a[i] != b[i]) return 0;
    return 1;
}

int main(int argc, char **argv) {
    /* Unbuffered so load/token lines reach gem5 logs before SE exit. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    const char *model_path = LLM_DEFAULT_MODEL;
    int max_tokens = 256;
    float temp = 0.8f;
    float top_p = 0.9f;
    int top_k = 0;
    int ctx = 2048;
    int n_threads = 0;
    int n_prefill = 0;
    int n_decode = 0;
    int enable_thinking = 0;
    int set_temp = 0, set_top_p = 0, set_top_k = 0;
    const char *tok_tables = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            temp = (float)atof(argv[++i]);
            set_temp = 1;
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            top_p = (float)atof(argv[++i]);
            set_top_p = 1;
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            top_k = atoi(argv[++i]);
            set_top_k = 1;
        } else if (strcmp(argv[i], "--ctx") == 0 && i + 1 < argc) ctx = atoi(argv[++i]);
        else if (strcmp(argv[i], "--think") == 0) enable_thinking = 1;
        else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) n_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--prefill-threads") == 0 && i + 1 < argc) n_prefill = atoi(argv[++i]);
        else if (strcmp(argv[i], "--decode-threads") == 0 && i + 1 < argc) n_decode = atoi(argv[++i]);
        else if (strcmp(argv[i], "--tok-tables") == 0 && i + 1 < argc) tok_tables = argv[++i];
        else if (strcmp(argv[i], "--quiet") == 0) llm_set_quiet(1);
        else if (strcmp(argv[i], "--steps") == 0) llm_set_quiet(0);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    if (n_threads > 0) llm_backend_set_threads(n_threads);
    if (n_prefill > 0) llm_backend_set_prefill_threads(n_prefill);
    if (n_decode > 0) llm_backend_set_decode_threads(n_decode);

    {
        const char *q = getenv("LLM_QUIET");
        if (q && q[0] && q[0] != '0') llm_set_quiet(1);
    }

    if (llm_steps_enabled()) {
        printf("loading %s\n", model_path);
        int np = llm_backend_n_prefill_threads();
        int nd = llm_backend_n_decode_threads();
        if (np == nd) printf("  backend %s, %d thread(s)\n", llm_backend_name(), np);
        else printf("  backend %s, %d prefill / %d decode thread(s)\n", llm_backend_name(), np, nd);
        fflush(stdout);
    }
    LoadedModel *loaded = load_model(model_path, 1, llm_steps_enabled());
    LLM_STEP("weights loaded, init model graph\n");
    LlamaModel *model = llama_model_init(loaded);
    LLM_STEP("model init done layers=%d embd=%d vocab=%d\n",
             model->hparams.n_layer, model->hparams.n_embd, model->hparams.n_vocab);
    LLM_STEP("build tokenizer tables=%s\n", tok_tables ? tok_tables : "(in-guest)");
    Tokenizer *tok = tokenizer_from_gguf_ex(loaded->gguf, &loaded->hparams, tok_tables);
    LLM_STEP("tokenizer ready vocab=%d\n", tok->n_vocab);
    tok->hparams.enable_thinking = enable_thinking;
    if (tok->hparams.arch == LLM_ARCH_QWEN2 || tok->hparams.arch == LLM_ARCH_QWEN35) {
        /* Qwen3 non-thinking defaults. Unconstrained 0.8/0.9 sampling repeats and derails. */
        if (!set_temp) temp = 0.7f;
        if (!set_top_p) top_p = 0.8f;
        if (!set_top_k) top_k = 20;
    }
    LLM_STEP("alloc kv cache ctx=%d\n", ctx);
    KVCache *cache = llama_model_new_cache(model, 1, ctx);
    LLM_STEP("kv cache ready\n");

    ChatMessage *history = NULL;
    int nh = 0, hcap = 0;
    IntVec cached_ids;
    intvec_init(&cached_ids);

    printf("Chat ready (temp=%.2f top-p=%.2f top-k=%d). Empty line or /exit to quit, /reset to clear history.\n\n",
           temp, top_p, top_k);
    char *line = NULL;
    size_t linecap = 0;
    for (;;) {
        fputs("You: ", stdout);
        fflush(stdout);
        ssize_t nr = getline(&line, &linecap, stdin);
        if (nr < 0) {
            fputc('\n', stdout);
            break;
        }
        while (nr > 0 && (line[nr - 1] == '\n' || line[nr - 1] == '\r')) line[--nr] = 0;
        char *user = line;
        while (*user == ' ' || *user == '\t') user++;
        size_t ulen = strlen(user);
        while (ulen && (user[ulen - 1] == ' ' || user[ulen - 1] == '\t')) user[--ulen] = 0;
        if (!user[0] || strcmp(user, "/exit") == 0 || strcmp(user, "/quit") == 0) break;
        if (strcmp(user, "/reset") == 0) {
            for (int i = 0; i < nh; i++) {
                free(history[i].role);
                free(history[i].content);
            }
            nh = 0;
            intvec_clear(&cached_ids);
            kvcache_reset(cache);
            printf("(context cleared)\n");
            continue;
        }

        double t0 = monotonic_now();
        if (nh >= hcap) {
            hcap = hcap ? hcap * 2 : 8;
            history = xrealloc(history, (size_t)hcap * sizeof(ChatMessage));
        }
        history[nh].role = xstrdup("user");
        history[nh].content = xstrdup(user);
        nh++;

        char *full = apply_chat_template(tok, history, nh, 1);
        IntVec ids;
        intvec_init(&ids);
        tokenizer_encode(tok, full, 1, &ids);
        free(full);

        const int *prompt_ids = ids.data;
        int n_prompt = ids.n;
        if (cached_ids.n && prefix_equal(cached_ids.data, cached_ids.n, ids.data, ids.n)) {
            prompt_ids = ids.data + cached_ids.n;
            n_prompt = ids.n - cached_ids.n;
        } else {
            kvcache_reset(cache);
            intvec_clear(&cached_ids);
        }
        if (n_prompt <= 0) {
            intvec_free(&ids);
            continue;
        }

        LLM_STEP("chat turn prompt_tokens=%d max_tokens=%d\n", n_prompt, max_tokens);
        fputs("Assistant: ", stdout);
        fflush(stdout);
        StreamDecoder dec;
        stream_decoder_init(&dec, tok, 1);
        IntVec gen_ids;
        intvec_init(&gen_ids);
        GenerateState st;
        generate_state_init(&st, model, tok, cache, max_tokens, temp, top_k, top_p, 0);
        generate_start(&st, prompt_ids, n_prompt);
        int tid;
        while (generate_next(&st, &tid) == 0) {
            intvec_push(&gen_ids, tid);
            char *chunk = stream_decoder_push(&dec, tid);
            if (chunk[0]) {
                fputs(chunk, stdout);
                fflush(stdout);
            }
            free(chunk);
        }
        char *tail = stream_decoder_flush(&dec);
        if (tail[0]) {
            fputs(tail, stdout);
            fflush(stdout);
        }
        free(tail);
        double elapsed = monotonic_now() - t0;
        double tps = elapsed > 0.0 ? (double)gen_ids.n / elapsed : 0.0;
        printf("\n[%d tokens, %.2f tok/s]\n", gen_ids.n, tps);
        fflush(stdout);
        generate_state_free(&st);
        stream_decoder_free(&dec);

        char *reply = tokenizer_decode(tok, gen_ids.data, gen_ids.n, 1);
        size_t rlen = strlen(reply);
        while (rlen && (reply[rlen - 1] == ' ' || reply[rlen - 1] == '\n' || reply[rlen - 1] == '\t'))
            reply[--rlen] = 0;
        strip_end_marker(reply);
        if (nh >= hcap) {
            hcap = hcap ? hcap * 2 : 8;
            history = xrealloc(history, (size_t)hcap * sizeof(ChatMessage));
        }
        history[nh].role = xstrdup("assistant");
        history[nh].content = reply;
        nh++;

        intvec_clear(&cached_ids);
        intvec_extend(&cached_ids, ids.data, ids.n);
        intvec_extend(&cached_ids, gen_ids.data, gen_ids.n);
        intvec_free(&ids);
        intvec_free(&gen_ids);
    }
    free(line);
    for (int i = 0; i < nh; i++) {
        free(history[i].role);
        free(history[i].content);
    }
    free(history);
    intvec_free(&cached_ids);
    kvcache_free(cache);
    tokenizer_free(tok);
    llama_model_free(model);
    loaded_model_free(loaded);
    return 0;
}
