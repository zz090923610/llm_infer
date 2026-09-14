#include "tokenizer.h"
#include "heap.h"
#include "unicode.h"

#define TYPE_UNKNOWN 2
#define TYPE_CONTROL 3
#define TYPE_USER_DEFINED 4

static uint32_t byte_to_cp[256];
static int cp_to_byte[512];
static int byte_tables_ready;

static void init_byte_tables(void) {
    if (byte_tables_ready) return;
    int bs[256];
    int cs[256];
    int n = 0;
    for (int b = 33; b <= 126; b++) bs[n++] = b;
    for (int b = 161; b <= 172; b++) bs[n++] = b;
    for (int b = 174; b <= 255; b++) bs[n++] = b;
    for (int i = 0; i < n; i++) cs[i] = bs[i];
    int in_bs[256] = {0};
    for (int i = 0; i < n; i++) in_bs[bs[i]] = 1;
    int extra = 0;
    for (int b = 0; b < 256; b++) {
        if (!in_bs[b]) {
            bs[n] = b;
            cs[n] = 256 + extra;
            n++;
            extra++;
        }
    }
    memset(cp_to_byte, 0xff, sizeof(cp_to_byte));
    for (int i = 0; i < 256; i++) {
        byte_to_cp[bs[i]] = (uint32_t)cs[i];
        if (cs[i] >= 0 && cs[i] < 512) cp_to_byte[cs[i]] = bs[i];
    }
    byte_tables_ready = 1;
}

static char *cps_slice_utf8(const uint32_t *cps, int start, int end) {
    ByteVec v;
    bytevec_init(&v);
    char tmp[4];
    for (int i = start; i < end; i++) {
        int n = utf8_encode(cps[i], tmp);
        bytevec_append(&v, tmp, n);
    }
    bytevec_push(&v, 0);
    return v.data;
}

static void gpt2_split(const uint32_t *chars, int n, char ***out, int *n_out) {
    char **words = NULL;
    int nw = 0, cap = 0;
    int pos = 0;
    while (pos < n) {
        uint32_t ch = chars[pos];
        if (ch == '\'' && pos + 1 < n) {
            uint32_t nxt = chars[pos + 1];
            if (nxt == 's' || nxt == 't' || nxt == 'm' || nxt == 'd') {
                if (nw >= cap) {
                    cap = cap ? cap * 2 : 8;
                    words = xrealloc(words, (size_t)cap * sizeof(char *));
                }
                words[nw++] = cps_slice_utf8(chars, pos, pos + 2);
                pos += 2;
                continue;
            }
            if (pos + 2 < n) {
                uint32_t a = nxt, b = chars[pos + 2];
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) {
                    if (nw >= cap) {
                        cap = cap ? cap * 2 : 8;
                        words = xrealloc(words, (size_t)cap * sizeof(char *));
                    }
                    words[nw++] = cps_slice_utf8(chars, pos, pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }
        int look_i = (ch == ' ' && pos + 1 < n) ? pos + 1 : pos;
        uint32_t look = look_i < n ? chars[look_i] : 0;
        if (look && is_letter(look)) {
            int start = pos;
            pos = look_i;
            while (pos < n && is_letter(chars[pos])) pos++;
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(chars, start, pos);
            continue;
        }
        if (look && is_number(look)) {
            int start = pos;
            pos = look_i;
            while (pos < n && is_number(chars[pos])) pos++;
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(chars, start, pos);
            continue;
        }
        if (look && !is_space(look) && !is_letter(look) && !is_number(look)) {
            int start = pos;
            pos = look_i;
            while (pos < n && !is_space(chars[pos]) && !is_letter(chars[pos]) && !is_number(chars[pos]))
                pos++;
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(chars, start, pos);
            continue;
        }
        int ws = 0;
        while (pos + ws < n && is_space(chars[pos + ws])) ws++;
        if (ws > 1 && pos + ws < n) {
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(chars, pos, pos + ws - 1);
            pos += ws - 1;
            continue;
        }
        if (ws > 0) {
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(chars, pos, pos + ws);
            pos += ws;
            continue;
        }
        if (nw >= cap) {
            cap = cap ? cap * 2 : 8;
            words = xrealloc(words, (size_t)cap * sizeof(char *));
        }
        words[nw++] = cps_slice_utf8(chars, pos, pos + 1);
        pos += 1;
    }
    *out = words;
    *n_out = nw;
}

static void pretokenize_smollm(const char *text, char ***out, int *n_out) {
    int ncp = 0;
    uint32_t *cps = utf8_codepoints(text, &ncp);
    char **words = NULL;
    int nw = 0, cap = 0;
    uint32_t *buf = xmalloc((size_t)(ncp + 1) * sizeof(uint32_t));
    int nb = 0;
    for (int i = 0; i < ncp; i++) {
        if (is_number(cps[i])) {
            if (nb) {
                char **part = NULL;
                int np = 0;
                gpt2_split(buf, nb, &part, &np);
                for (int j = 0; j < np; j++) {
                    if (nw >= cap) {
                        cap = cap ? cap * 2 : 8;
                        words = xrealloc(words, (size_t)cap * sizeof(char *));
                    }
                    words[nw++] = part[j];
                }
                free(part);
                nb = 0;
            }
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(cps, i, i + 1);
        } else {
            buf[nb++] = cps[i];
        }
    }
    if (nb) {
        char **part = NULL;
        int np = 0;
        gpt2_split(buf, nb, &part, &np);
        for (int j = 0; j < np; j++) {
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = part[j];
        }
        free(part);
    }
    free(buf);
    free(cps);
    *out = words;
    *n_out = nw;
}

static uint32_t ascii_tolower(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') return cp - 'A' + 'a';
    return cp;
}

static int is_letter_run(uint32_t cp, int with_marks) {
    return is_letter(cp) || (with_marks && is_mark(cp));
}

/* Qwen2 / Qwen3.5 BPE pretok (llama.cpp unicode_regex_split_custom_qwen*). */
static void pretokenize_qwen(const char *text, int with_marks, char ***out, int *n_out) {
    int ncp = 0;
    uint32_t *cps = utf8_codepoints(text, &ncp);
    char **words = NULL;
    int nw = 0, cap = 0;
    const uint32_t OOR = 0xFFFFFFFFu;
    int pos = 0;
    while (pos < ncp) {
        uint32_t cpt = cps[pos];
        if (cpt == '\'' && pos + 1 < ncp) {
            uint32_t nxt = ascii_tolower(cps[pos + 1]);
            if (nxt == 's' || nxt == 't' || nxt == 'm' || nxt == 'd') {
                if (nw >= cap) {
                    cap = cap ? cap * 2 : 8;
                    words = xrealloc(words, (size_t)cap * sizeof(char *));
                }
                words[nw++] = cps_slice_utf8(cps, pos, pos + 2);
                pos += 2;
                continue;
            }
            if (pos + 2 < ncp) {
                uint32_t nn = ascii_tolower(cps[pos + 2]);
                if ((nxt == 'r' && nn == 'e') || (nxt == 'v' && nn == 'e') || (nxt == 'l' && nn == 'l')) {
                    if (nw >= cap) {
                        cap = cap ? cap * 2 : 8;
                        words = xrealloc(words, (size_t)cap * sizeof(char *));
                    }
                    words[nw++] = cps_slice_utf8(cps, pos, pos + 3);
                    pos += 3;
                    continue;
                }
            }
        }
        /* [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+  (qwen2: letters only in the run) */
        if (!(cpt == '\r' || cpt == '\n' || is_number(cpt))) {
            uint32_t nxt = (pos + 1 < ncp) ? cps[pos + 1] : OOR;
            int nxt_run = (nxt != OOR) && is_letter_run(nxt, with_marks);
            if (is_letter_run(cpt, with_marks) || nxt_run) {
                int start = pos;
                pos++;
                while (pos < ncp && is_letter_run(cps[pos], with_marks)) pos++;
                if (nw >= cap) {
                    cap = cap ? cap * 2 : 8;
                    words = xrealloc(words, (size_t)cap * sizeof(char *));
                }
                words[nw++] = cps_slice_utf8(cps, start, pos);
                continue;
            }
        }
        if (is_number(cpt)) {
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(cps, pos, pos + 1);
            pos++;
            continue;
        }
        /* optional space + punctuation + newlines */
        {
            int look = (cpt == ' ' && pos + 1 < ncp) ? pos + 1 : pos;
            uint32_t lookc = look < ncp ? cps[look] : 0;
            int punct = look < ncp && lookc && !is_space(lookc) && !is_letter(lookc) && !is_number(lookc) &&
                        !(with_marks && is_mark(lookc));
            if (punct) {
                int start = pos;
                pos = look;
                while (pos < ncp) {
                    uint32_t c = cps[pos];
                    if (is_space(c) || is_letter(c) || is_number(c) || (with_marks && is_mark(c))) break;
                    pos++;
                }
                while (pos < ncp && (cps[pos] == '\r' || cps[pos] == '\n')) pos++;
                if (nw >= cap) {
                    cap = cap ? cap * 2 : 8;
                    words = xrealloc(words, (size_t)cap * sizeof(char *));
                }
                words[nw++] = cps_slice_utf8(cps, start, pos);
                continue;
            }
        }
        int num_ws = 0;
        int last_rn = 0;
        while (pos + num_ws < ncp && is_space(cps[pos + num_ws])) {
            uint32_t c = cps[pos + num_ws];
            if (c == '\r' || c == '\n') last_rn = pos + num_ws + 1;
            num_ws++;
        }
        if (last_rn > 0) {
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(cps, pos, last_rn);
            pos = last_rn;
            continue;
        }
        if (num_ws > 1 && pos + num_ws < ncp) {
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(cps, pos, pos + num_ws - 1);
            pos += num_ws - 1;
            continue;
        }
        if (num_ws > 0) {
            if (nw >= cap) {
                cap = cap ? cap * 2 : 8;
                words = xrealloc(words, (size_t)cap * sizeof(char *));
            }
            words[nw++] = cps_slice_utf8(cps, pos, pos + num_ws);
            pos += num_ws;
            continue;
        }
        if (nw >= cap) {
            cap = cap ? cap * 2 : 8;
            words = xrealloc(words, (size_t)cap * sizeof(char *));
        }
        words[nw++] = cps_slice_utf8(cps, pos, pos + 1);
        pos++;
    }
    free(cps);
    *out = words;
    *n_out = nw;
}

static int str_has_ci(const char *s, const char *needle) {
    if (!s || !needle || !needle[0]) return 0;
    for (; *s; s++) {
        const char *a = s, *b = needle;
        while (*a && *b) {
            char ca = *a, cb = *b;
            if (ca >= 'A' && ca <= 'Z') ca = (char)(ca - 'A' + 'a');
            if (cb >= 'A' && cb <= 'Z') cb = (char)(cb - 'A' + 'a');
            if (ca != cb) break;
            a++;
            b++;
        }
        if (!*b) return 1;
    }
    return 0;
}

char *apply_chat_template(const Tokenizer *tok, const ChatMessage *msgs, int n, int add_generation_prompt) {
    ByteVec v;
    bytevec_init(&v);
    int arch = tok ? tok->hparams.arch : LLM_ARCH_LLAMA;
    int has_system = n > 0 && msgs[0].role && strcmp(msgs[0].role, "system") == 0;
    if (!has_system) {
        /* Qwen2.5 jinja injects this when no system message is given. Qwen3.5's
           template does not; the extra "helpful assistant" clause is what 9B
           was parroting. MiniCPM shares qwen35 and must not get a Qwen identity. */
        const char *sys = NULL;
        if (arch == LLM_ARCH_QWEN2) {
            sys = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n";
        } else if (arch == LLM_ARCH_QWEN35 && tok && str_has_ci(tok->hparams.model_name, "Qwen")) {
            sys = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.<|im_end|>\n";
        } else if (tok && tok->hparams.chat_template && strstr(tok->hparams.chat_template, "SmolLM")) {
            /* Official HuggingFaceTB SmolLM2 jinja injects this when no system
               message is given. */
            sys = "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";
        }
        if (sys) bytevec_append(&v, sys, (int)strlen(sys));
    }
    for (int i = 0; i < n; i++) {
        const char *content = msgs[i].content ? msgs[i].content : "";
        while (*content == ' ' || *content == '\t' || *content == '\n' || *content == '\r') content++;
        int clen = (int)strlen(content);
        while (clen > 0 && (content[clen - 1] == ' ' || content[clen - 1] == '\t' ||
                            content[clen - 1] == '\n' || content[clen - 1] == '\r'))
            clen--;
        bytevec_append(&v, "<|im_start|>", 12);
        bytevec_append(&v, msgs[i].role, (int)strlen(msgs[i].role));
        bytevec_push(&v, '\n');
        /* Match the non-thinking generation prefix on prior assistant turns so
           the model does not see a format change mid-chat. */
        if (arch == LLM_ARCH_QWEN35 && msgs[i].role && strcmp(msgs[i].role, "assistant") == 0 &&
            !(tok && tok->hparams.enable_thinking)) {
            bytevec_append(&v, "<think>\n\n</think>\n\n", 19);
        }
        if (clen > 0) bytevec_append(&v, content, clen);
        bytevec_append(&v, "<|im_end|>\n", 11);
    }
    if (add_generation_prompt) {
        bytevec_append(&v, "<|im_start|>assistant\n", 22);
        if (arch == LLM_ARCH_QWEN35) {
            if (tok && tok->hparams.enable_thinking) bytevec_append(&v, "<think>\n", 8);
            else bytevec_append(&v, "<think>\n\n</think>\n\n", 19);
        }
    }
    bytevec_push(&v, 0);
    return v.data;
}

static int arr_int_at(const GGUFValue *v, int i, int def) {
    if (!v || v->type != GGUF_ARR || i < 0 || i >= v->v.arr.n) return def;
    const unsigned char *p = (const unsigned char *)v->v.arr.data;
    int t = v->v.arr.type;
    if (t == GGUF_I32 || t == GGUF_U32) {
        int32_t x;
        memcpy(&x, p + (size_t)i * 4, 4);
        return (int)x;
    }
    if (t == GGUF_I16 || t == GGUF_U16) {
        int16_t x;
        memcpy(&x, p + (size_t)i * 2, 2);
        return (int)x;
    }
    if (t == GGUF_I8 || t == GGUF_U8 || t == GGUF_BOOL) return (int)p[i];
    if (t == GGUF_I64 || t == GGUF_U64) {
        int64_t x;
        memcpy(&x, p + (size_t)i * 8, 8);
        return (int)x;
    }
    return def;
}

typedef struct {
    int id;
    const char *tok;
    int len;
} SpecRef;

static int spec_cmp(const void *a, const void *b) {
    const SpecRef *sa = a, *sb = b;
    if (sa->len != sb->len) return sb->len - sa->len;
    return sa->id - sb->id;
}

static int is_think_open(const char *tok) { return tok && strcmp(tok, "<think>") == 0; }
static int is_think_close(const char *tok) { return tok && strcmp(tok, "</think>") == 0; }

static int is_special_tok(const char *tok, int ttype) {
    if (is_think_open(tok) || is_think_close(tok)) return 1;
    if (ttype == TYPE_UNKNOWN || ttype == TYPE_CONTROL || ttype == TYPE_USER_DEFINED) return 1;
    size_t n = strlen(tok);
    return n >= 4 && tok[0] == '<' && tok[1] == '|' && tok[n - 2] == '|' && tok[n - 1] == '>';
}

static int merge_pair_cmp(const void *a, const void *b) {
    uint64_t ka = ((const MergePair *)a)->key;
    uint64_t kb = ((const MergePair *)b)->key;
    return (ka > kb) - (ka < kb);
}

enum { LLM_TOK_TABLE_MAGIC = 0x504D4C4Cu, LLM_TOK_TABLE_VERSION = 1 };

static void load_merge_table(Tokenizer *t, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die("cannot open --tok-tables %s", path);
    if (fseek(fp, 0, SEEK_END) != 0) die("fseek --tok-tables %s", path);
    long sz = ftell(fp);
    if (sz < 16) die("truncated --tok-tables %s", path);
    if (fseek(fp, 0, SEEK_SET) != 0) die("fseek --tok-tables %s", path);
    uint8_t *buf = xmalloc((size_t)sz);
    if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) die("read --tok-tables %s", path);
    fclose(fp);

    uint32_t magic, ver, n_vocab, n_pairs;
    memcpy(&magic, buf, 4);
    memcpy(&ver, buf + 4, 4);
    memcpy(&n_vocab, buf + 8, 4);
    memcpy(&n_pairs, buf + 12, 4);
    if (magic != LLM_TOK_TABLE_MAGIC) die("bad --tok-tables magic in %s", path);
    if (ver != LLM_TOK_TABLE_VERSION) die("unsupported --tok-tables version %u", ver);
    if ((int)n_vocab != t->n_vocab) {
        die("--tok-tables n_vocab=%u != model vocab %d", n_vocab, t->n_vocab);
    }
    size_t need = 16u + (size_t)n_pairs * 12u;
    if ((size_t)sz < need) die("truncated --tok-tables pairs in %s", path);

    t->merge_pairs = xmalloc((size_t)n_pairs * sizeof(MergePair));
    const uint8_t *p = buf + 16;
    for (uint32_t i = 0; i < n_pairs; i++) {
        uint64_t key;
        int32_t rank;
        memcpy(&key, p, 8);
        memcpy(&rank, p + 8, 4);
        p += 12;
        t->merge_pairs[i].key = key;
        t->merge_pairs[i].rank = (int)rank;
    }
    t->n_merge_pairs = (int)n_pairs;
    free(buf);
    LLM_STEP("tokenizer loaded %d merge pairs from %s\n", t->n_merge_pairs, path);
}

static void index_merges_from_gguf(Tokenizer *t, const GGUFValue *merges) {
    t->merge_pairs = xmalloc((size_t)merges->v.arr.n * sizeof(MergePair));
    t->n_merge_pairs = 0;
    LLM_STEP("tokenizer index merges as id pairs (slow in-guest path)\n");
    char **ms = (char **)merges->v.arr.data;
    int n_merge_skip = 0;
    for (int i = 0; i < merges->v.arr.n; i++) {
        char *sp = strchr(ms[i], ' ');
        if (!sp || sp == ms[i] || !sp[1]) {
            n_merge_skip++;
            continue;
        }
        char saved = *sp;
        *sp = 0;
        int lid = -1, rid = -1;
        int got_l = strmap_get(&t->token_to_id, ms[i], &lid);
        *sp = saved;
        int got_r = strmap_get(&t->token_to_id, sp + 1, &rid);
        if (!got_l || !got_r) {
            n_merge_skip++;
            continue;
        }
        t->merge_pairs[t->n_merge_pairs].key =
            ((uint64_t)(uint32_t)lid << 32) | (uint32_t)rid;
        t->merge_pairs[t->n_merge_pairs].rank = i;
        t->n_merge_pairs++;
        if ((i + 1) % 1024 == 0 || i + 1 == merges->v.arr.n) {
            LLM_STEP("tokenizer merges %d/%d (ok=%d)\n", i + 1, merges->v.arr.n,
                     t->n_merge_pairs);
        }
    }
    LLM_STEP("tokenizer qsort %d merge pairs\n", t->n_merge_pairs);
    qsort(t->merge_pairs, (size_t)t->n_merge_pairs, sizeof(MergePair), merge_pair_cmp);
    LLM_STEP("tokenizer merges done ok=%d skip=%d\n", t->n_merge_pairs, n_merge_skip);
}

Tokenizer *tokenizer_from_gguf(const GGUFFile *gguf, const LlamaHParams *hp) {
    return tokenizer_from_gguf_ex(gguf, hp, NULL);
}

Tokenizer *tokenizer_from_gguf_ex(const GGUFFile *gguf, const LlamaHParams *hp,
                                  const char *merge_table_path) {
    init_byte_tables();
    const GGUFValue *model = gguf_get(gguf, "tokenizer.ggml.model");
    if (!model || model->type != GGUF_STR || strcmp(model->v.s, "gpt2") != 0) {
        die("unsupported tokenizer model");
    }
    const GGUFValue *toks = gguf_get(gguf, "tokenizer.ggml.tokens");
    const GGUFValue *merges = gguf_get(gguf, "tokenizer.ggml.merges");
    const GGUFValue *types = gguf_get(gguf, "tokenizer.ggml.token_type");
    if (!toks || toks->type != GGUF_ARR || toks->v.arr.type != GGUF_STR) die("missing tokenizer tokens");
    int have_tables = merge_table_path && merge_table_path[0];
    if (!have_tables && (!merges || merges->type != GGUF_ARR || merges->v.arr.type != GGUF_STR)) {
        die("missing tokenizer merges");
    }

    Tokenizer *t = xcalloc(1, sizeof(Tokenizer));
    if (hp) {
        t->hparams = *hp;
        t->hparams.chat_template = hp->chat_template ? xstrdup(hp->chat_template) : NULL;
        t->hparams.tokenizer_pre = hp->tokenizer_pre ? xstrdup(hp->tokenizer_pre) : NULL;
        t->hparams.model_name = hp->model_name ? xstrdup(hp->model_name) : NULL;
    } else {
        hparams_from_kv(gguf, &t->hparams);
    }
    t->n_vocab = toks->v.arr.n;
    t->vocab = xcalloc((size_t)t->n_vocab, sizeof(char *));
    strmap_init(&t->token_to_id, t->n_vocab);
    LLM_STEP("tokenizer copy vocab n=%d buckets=%d\n", t->n_vocab, t->token_to_id.nbuckets);
    char **ss = (char **)toks->v.arr.data;
    for (int i = 0; i < t->n_vocab; i++) {
        t->vocab[i] = xstrdup(ss[i]);
        strmap_put(&t->token_to_id, t->vocab[i], i);
        if ((i + 1) % 4096 == 0 || i + 1 == t->n_vocab) {
            LLM_STEP("tokenizer vocab %d/%d\n", i + 1, t->n_vocab);
        }
    }
    LLM_STEP("tokenizer vocab done\n");

    if (have_tables) {
        load_merge_table(t, merge_table_path);
    } else {
        index_merges_from_gguf(t, merges);
    }

    SpecRef *refs = xmalloc((size_t)t->n_vocab * sizeof(SpecRef));
    int ns = 0;
    for (int i = 0; i < t->n_vocab; i++) {
        int ttype = arr_int_at(types, i, 1);
        if (is_special_tok(t->vocab[i], ttype)) {
            refs[ns].id = i;
            refs[ns].tok = t->vocab[i];
            refs[ns].len = (int)strlen(t->vocab[i]);
            ns++;
        }
    }
    qsort(refs, (size_t)ns, sizeof(SpecRef), spec_cmp);
    t->n_specials = ns;
    t->specials = xcalloc((size_t)ns, sizeof(char *));
    hashmap_init(&t->special_set);
    for (int i = 0; i < ns; i++) {
        t->specials[i] = xstrdup(refs[i].tok);
        hashmap_put(&t->special_set, t->specials[i], 1);
    }
    free(refs);
    LLM_STEP("tokenizer specials n=%d\n", t->n_specials);

    IntVec stops;
    intvec_init(&stops);
    intvec_push(&stops, t->hparams.eos_id);
    int tid;
    /* SmolLM2 uses literal END as EOS. Do not treat those pieces as stops on
       Qwen — they are ordinary BPE tokens and would truncate replies. */
    if (t->hparams.arch == LLM_ARCH_LLAMA) {
        if (strmap_get(&t->token_to_id, "END", &tid)) intvec_push(&stops, tid);
        if (strmap_get(&t->token_to_id, "ĠEND", &tid)) intvec_push(&stops, tid);
    }
    t->n_stop = stops.n;
    t->stop_ids = stops.data;
    return t;
}

Tokenizer *tokenizer_from_file(const char *path) {
    GGUFFile *f = open_gguf(path);
    LlamaHParams hp;
    hparams_from_kv(f, &hp);
    Tokenizer *t = tokenizer_from_gguf(f, &hp);
    hparams_free(&hp);
    gguf_close(f);
    return t;
}

void tokenizer_free(Tokenizer *t) {
    if (!t) return;
    for (int i = 0; i < t->n_vocab; i++) free(t->vocab[i]);
    free(t->vocab);
    strmap_free(&t->token_to_id);
    free(t->merge_pairs);
    for (int i = 0; i < t->n_specials; i++) free(t->specials[i]);
    free(t->specials);
    hashmap_free(&t->special_set);
    hparams_free(&t->hparams);
    free(t->stop_ids);
    free(t);
}

typedef struct {
    int special;
    char *s;
} Frag;

static void partition(Tokenizer *t, const char *text, int parse_special, Frag **out, int *n_out) {
    Frag *frags = xmalloc(sizeof(Frag));
    frags[0].special = 0;
    frags[0].s = xstrdup(text);
    int nf = 1;
    if (!parse_special || t->n_specials == 0) {
        if (frags[0].s[0] == '\0') {
            free(frags[0].s);
            free(frags);
            *out = NULL;
            *n_out = 0;
            return;
        }
        *out = frags;
        *n_out = 1;
        return;
    }
    for (int si = 0; si < t->n_specials; si++) {
        const char *spec = t->specials[si];
        size_t slen = strlen(spec);
        Frag *nxt = NULL;
        int nn = 0, cap = 0;
        for (int fi = 0; fi < nf; fi++) {
            if (frags[fi].special || !strstr(frags[fi].s, spec)) {
                if (nn >= cap) {
                    cap = cap ? cap * 2 : 8;
                    nxt = xrealloc(nxt, (size_t)cap * sizeof(Frag));
                }
                nxt[nn++] = frags[fi];
                frags[fi].s = NULL;
                continue;
            }
            char *piece = frags[fi].s;
            char *p = piece;
            while (*p) {
                char *hit = strstr(p, spec);
                if (!hit) {
                    if (*p) {
                        if (nn >= cap) {
                            cap = cap ? cap * 2 : 8;
                            nxt = xrealloc(nxt, (size_t)cap * sizeof(Frag));
                        }
                        nxt[nn].special = 0;
                        nxt[nn].s = xstrdup(p);
                        nn++;
                    }
                    break;
                }
                if (hit > p) {
                    if (nn >= cap) {
                        cap = cap ? cap * 2 : 8;
                        nxt = xrealloc(nxt, (size_t)cap * sizeof(Frag));
                    }
                    nxt[nn].special = 0;
                    nxt[nn].s = xstrndup(p, (size_t)(hit - p));
                    nn++;
                }
                if (nn >= cap) {
                    cap = cap ? cap * 2 : 8;
                    nxt = xrealloc(nxt, (size_t)cap * sizeof(Frag));
                }
                nxt[nn].special = 1;
                nxt[nn].s = xstrdup(spec);
                nn++;
                p = hit + slen;
            }
            free(piece);
        }
        free(frags);
        frags = nxt;
        nf = nn;
    }
    int w = 0;
    for (int i = 0; i < nf; i++) {
        if (frags[i].s && frags[i].s[0]) frags[w++] = frags[i];
        else free(frags[i].s);
    }
    *out = frags;
    *n_out = w;
}

static int merge_rank(Tokenizer *t, const char *left, const char *right, int *rank) {
    int lid, rid;
    if (!strmap_get(&t->token_to_id, left, &lid)) return 0;
    if (!strmap_get(&t->token_to_id, right, &rid)) return 0;
    MergePair key;
    key.key = ((uint64_t)(uint32_t)lid << 32) | (uint32_t)rid;
    key.rank = 0;
    const MergePair *hit =
        bsearch(&key, t->merge_pairs, (size_t)t->n_merge_pairs, sizeof(MergePair), merge_pair_cmp);
    if (!hit) return 0;
    if (rank) *rank = hit->rank;
    return 1;
}

static void bpe(Tokenizer *t, const char *token, IntVec *ids) {
    int tid;
    if (strmap_get(&t->token_to_id, token, &tid)) {
        intvec_push(ids, tid);
        return;
    }
    int ncp = 0;
    uint32_t *cps = utf8_codepoints(token, &ncp);
    if (ncp == 0) {
        free(cps);
        return;
    }
    if (ncp == 1) {
        char tmp[8];
        int n = utf8_encode(cps[0], tmp);
        tmp[n] = 0;
        if (strmap_get(&t->token_to_id, tmp, &tid)) intvec_push(ids, tid);
        free(cps);
        return;
    }

    char **word = xmalloc((size_t)ncp * sizeof(char *));
    int *alive = xmalloc((size_t)ncp * sizeof(int));
    int *next_idx = xmalloc((size_t)ncp * sizeof(int));
    int *prev_idx = xmalloc((size_t)ncp * sizeof(int));
    for (int i = 0; i < ncp; i++) {
        char tmp[8];
        int n = utf8_encode(cps[i], tmp);
        tmp[n] = 0;
        word[i] = xstrdup(tmp);
        alive[i] = 1;
        next_idx[i] = (i + 1 < ncp) ? i + 1 : -1;
        prev_idx[i] = i - 1;
    }
    free(cps);

    MinHeap heap;
    heap_init(&heap);
    for (int i = 0; i < ncp - 1; i++) {
        int r;
        if (merge_rank(t, word[i], word[i + 1], &r)) heap_push(&heap, r, i, i + 1);
    }

    HeapItem it;
    while (heap_pop(&heap, &it)) {
        int left = it.left, right = it.right, rank = it.rank;
        if (!alive[left] || !alive[right]) continue;
        if (next_idx[left] != right) continue;
        int cur;
        if (!merge_rank(t, word[left], word[right], &cur) || cur != rank) continue;
        size_t ll = strlen(word[left]), lr = strlen(word[right]);
        char *merged = xmalloc(ll + lr + 1);
        memcpy(merged, word[left], ll);
        memcpy(merged + ll, word[right], lr + 1);
        free(word[left]);
        word[left] = merged;
        alive[right] = 0;
        int nxt = next_idx[right];
        next_idx[left] = nxt;
        if (nxt != -1) prev_idx[nxt] = left;
        int prv = prev_idx[left];
        if (prv != -1) {
            int r;
            if (merge_rank(t, word[prv], word[left], &r)) heap_push(&heap, r, prv, left);
        }
        if (nxt != -1) {
            int r;
            if (merge_rank(t, word[left], word[nxt], &r)) heap_push(&heap, r, left, nxt);
        }
    }
    heap_free(&heap);

    int i = 0;
    while (i != -1) {
        if (alive[i]) {
            if (strmap_get(&t->token_to_id, word[i], &tid)) {
                intvec_push(ids, tid);
            } else {
                int npc = 0;
                uint32_t *pc = utf8_codepoints(word[i], &npc);
                for (int k = 0; k < npc; k++) {
                    char tmp[8];
                    int n = utf8_encode(pc[k], tmp);
                    tmp[n] = 0;
                    if (strmap_get(&t->token_to_id, tmp, &tid)) intvec_push(ids, tid);
                }
                free(pc);
            }
        }
        i = next_idx[i];
    }
    for (int j = 0; j < ncp; j++) free(word[j]);
    free(word);
    free(alive);
    free(next_idx);
    free(prev_idx);
}

static char *gpt2_byte_encode(const char *word) {
    init_byte_tables();
    ByteVec v;
    bytevec_init(&v);
    for (const unsigned char *p = (const unsigned char *)word; *p; p++) {
        char tmp[4];
        int n = utf8_encode(byte_to_cp[*p], tmp);
        bytevec_append(&v, tmp, n);
    }
    bytevec_push(&v, 0);
    return v.data;
}

static void encode_text(Tokenizer *t, const char *text, IntVec *ids) {
    char **words = NULL;
    int nw = 0;
    const char *pre = t->hparams.tokenizer_pre ? t->hparams.tokenizer_pre : "";
    if (strcmp(pre, "qwen2") == 0) pretokenize_qwen(text, 0, &words, &nw);
    else if (strcmp(pre, "qwen35") == 0) pretokenize_qwen(text, 1, &words, &nw);
    else pretokenize_smollm(text, &words, &nw);
    for (int i = 0; i < nw; i++) {
        char *enc = gpt2_byte_encode(words[i]);
        bpe(t, enc, ids);
        free(enc);
        free(words[i]);
    }
    free(words);
}

int tokenizer_encode(Tokenizer *t, const char *text, int parse_special, IntVec *out) {
    Frag *frags = NULL;
    int nf = 0;
    partition(t, text, parse_special, &frags, &nf);
    for (int i = 0; i < nf; i++) {
        if (frags[i].special) {
            int tid;
            if (!strmap_get(&t->token_to_id, frags[i].s, &tid)) die("unknown special %s", frags[i].s);
            intvec_push(out, tid);
        } else {
            encode_text(t, frags[i].s, out);
        }
        free(frags[i].s);
    }
    free(frags);
    return 0;
}

static void decode_piece_bytes(Tokenizer *t, const char *piece, ByteVec *raw) {
    init_byte_tables();
    if (hashmap_get(&t->special_set, piece, NULL)) {
        bytevec_append(raw, piece, (int)strlen(piece));
        return;
    }
    int ncp = 0;
    uint32_t *cps = utf8_codepoints(piece, &ncp);
    for (int i = 0; i < ncp; i++) {
        uint32_t cp = cps[i];
        int b = (cp < 512) ? cp_to_byte[cp] : -1;
        if (b < 0) die("unknown byte decoder codepoint %u", cp);
        bytevec_push(raw, (unsigned char)b);
    }
    free(cps);
}

char *tokenizer_decode(Tokenizer *t, const int *ids, int n, int skip_special) {
    ByteVec raw;
    bytevec_init(&raw);
    int hide = skip_special && t->hparams.arch == LLM_ARCH_QWEN35 && t->hparams.enable_thinking;
    for (int i = 0; i < n; i++) {
        const char *piece = t->vocab[ids[i]];
        if (skip_special && is_think_open(piece)) {
            hide = 1;
            continue;
        }
        if (skip_special && is_think_close(piece)) {
            hide = 0;
            continue;
        }
        if (hide) continue;
        if (skip_special && hashmap_get(&t->special_set, piece, NULL)) continue;
        decode_piece_bytes(t, piece, &raw);
    }
    bytevec_push(&raw, 0);
    return raw.data;
}

static int utf8_valid_prefix(const char *s, int n) {
    size_t i = 0;
    while (i < (size_t)n) {
        unsigned char c = (unsigned char)s[i];
        int need;
        uint32_t cp;
        if (c < 0x80) {
            need = 1;
            cp = c;
        } else if ((c & 0xe0) == 0xc0) {
            need = 2;
            if (i + 1 >= (size_t)n) return -1;
            if (((unsigned char)s[i + 1] & 0xc0) != 0x80) return 0;
            cp = ((uint32_t)(c & 0x1f) << 6) | (uint32_t)(s[i + 1] & 0x3f);
            if (cp < 0x80) return 0;
        } else if ((c & 0xf0) == 0xe0) {
            need = 3;
            if (i + 2 >= (size_t)n) return -1;
            if (((unsigned char)s[i + 1] & 0xc0) != 0x80 || ((unsigned char)s[i + 2] & 0xc0) != 0x80) return 0;
            cp = ((uint32_t)(c & 0x0f) << 12) | ((uint32_t)(s[i + 1] & 0x3f) << 6) | (uint32_t)(s[i + 2] & 0x3f);
            if (cp < 0x800) return 0;
        } else if ((c & 0xf8) == 0xf0) {
            need = 4;
            if (i + 3 >= (size_t)n) return -1;
            if (((unsigned char)s[i + 1] & 0xc0) != 0x80 || ((unsigned char)s[i + 2] & 0xc0) != 0x80 ||
                ((unsigned char)s[i + 3] & 0xc0) != 0x80)
                return 0;
            cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[i + 1] & 0x3f) << 12) |
                 ((uint32_t)(s[i + 2] & 0x3f) << 6) | (uint32_t)(s[i + 3] & 0x3f);
            if (cp < 0x10000 || cp > 0x10ffff) return 0;
        } else {
            return 0;
        }
        (void)cp;
        i += (size_t)need;
    }
    return 1;
}

void stream_decoder_init(StreamDecoder *d, Tokenizer *t, int skip_special) {
    d->tok = t;
    d->skip_special = skip_special;
    /* Prompt already opened <think> when thinking is enabled. */
    d->in_think = t && t->hparams.arch == LLM_ARCH_QWEN35 && t->hparams.enable_thinking;
    bytevec_init(&d->buf);
}

void stream_decoder_free(StreamDecoder *d) { bytevec_free(&d->buf); }

char *stream_decoder_push(StreamDecoder *d, int token_id) {
    const char *piece = d->tok->vocab[token_id];
    if (is_think_open(piece)) {
        d->in_think = 1;
        return xstrdup("");
    }
    if (is_think_close(piece)) {
        d->in_think = 0;
        return xstrdup("");
    }
    if (d->in_think) return xstrdup("");
    if (d->skip_special && hashmap_get(&d->tok->special_set, piece, NULL)) return xstrdup("");
    decode_piece_bytes(d->tok, piece, &d->buf);
    if (utf8_valid_prefix(d->buf.data, d->buf.n) == 1) {
        char *s = xstrndup(d->buf.data, (size_t)d->buf.n);
        bytevec_clear(&d->buf);
        return s;
    }
    for (int drop = 1; drop <= 3; drop++) {
        if (d->buf.n <= drop) continue;
        if (utf8_valid_prefix(d->buf.data, d->buf.n - drop) == 1) {
            char *s = xstrndup(d->buf.data, (size_t)(d->buf.n - drop));
            memmove(d->buf.data, d->buf.data + d->buf.n - drop, (size_t)drop);
            d->buf.n = drop;
            return s;
        }
    }
    return xstrdup("");
}

char *stream_decoder_flush(StreamDecoder *d) {
    /* errors=replace: emit what we can, U+FFFD for leftovers */
    ByteVec out;
    bytevec_init(&out);
    int i = 0;
    while (i < d->buf.n) {
        int rem = d->buf.n - i;
        int ok = 0;
        for (int n = rem < 4 ? rem : 4; n >= 1; n--) {
            if (utf8_valid_prefix(d->buf.data + i, n) == 1) {
                bytevec_append(&out, d->buf.data + i, n);
                i += n;
                ok = 1;
                break;
            }
        }
        if (!ok) {
            char repl[4];
            int n = utf8_encode(0xfffd, repl);
            bytevec_append(&out, repl, n);
            i++;
        }
    }
    bytevec_clear(&d->buf);
    bytevec_push(&out, 0);
    return out.data;
}
