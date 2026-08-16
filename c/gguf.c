#include "gguf.h"
#include "quant.h"
#include "util.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static uint8_t rd_u8(const unsigned char *buf, size_t *pos) {
    return buf[(*pos)++];
}

static uint16_t rd_u16(const unsigned char *buf, size_t *pos) {
    uint16_t v;
    memcpy(&v, buf + *pos, 2);
    *pos += 2;
    return v;
}

static uint32_t rd_u32(const unsigned char *buf, size_t *pos) {
    uint32_t v;
    memcpy(&v, buf + *pos, 4);
    *pos += 4;
    return v;
}

static uint64_t rd_u64(const unsigned char *buf, size_t *pos) {
    uint64_t v;
    memcpy(&v, buf + *pos, 8);
    *pos += 8;
    return v;
}

static int8_t rd_i8(const unsigned char *buf, size_t *pos) {
    return (int8_t)rd_u8(buf, pos);
}

static int16_t rd_i16(const unsigned char *buf, size_t *pos) {
    return (int16_t)rd_u16(buf, pos);
}

static int32_t rd_i32(const unsigned char *buf, size_t *pos) {
    return (int32_t)rd_u32(buf, pos);
}

static int64_t rd_i64(const unsigned char *buf, size_t *pos) {
    return (int64_t)rd_u64(buf, pos);
}

static float rd_f32(const unsigned char *buf, size_t *pos) {
    float v;
    memcpy(&v, buf + *pos, 4);
    *pos += 4;
    return v;
}

static double rd_f64(const unsigned char *buf, size_t *pos) {
    double v;
    memcpy(&v, buf + *pos, 8);
    *pos += 8;
    return v;
}

static char *rd_str(const unsigned char *buf, size_t *pos, size_t map_size) {
    uint64_t n = rd_u64(buf, pos);
    if (*pos + n > map_size) die("GGUF string overruns file");
    char *s = xstrndup((const char *)(buf + *pos), (size_t)n);
    *pos += (size_t)n;
    return s;
}

static size_t scalar_size(int type) {
    switch (type) {
    case GGUF_U8:
    case GGUF_I8:
    case GGUF_BOOL:
        return 1;
    case GGUF_U16:
    case GGUF_I16:
        return 2;
    case GGUF_U32:
    case GGUF_I32:
    case GGUF_F32:
        return 4;
    case GGUF_U64:
    case GGUF_I64:
    case GGUF_F64:
        return 8;
    default:
        return 0;
    }
}

static void read_value(const unsigned char *buf, size_t *pos, size_t map_size, int vtype, GGUFValue *out);

static void read_array(const unsigned char *buf, size_t *pos, size_t map_size, GGUFValue *out) {
    int atype = (int)rd_u32(buf, pos);
    uint64_t n = rd_u64(buf, pos);
    out->type = GGUF_ARR;
    out->v.arr.type = atype;
    out->v.arr.n = (int)n;
    if (atype == GGUF_STR) {
        char **ss = xcalloc((size_t)n, sizeof(char *));
        for (uint64_t i = 0; i < n; i++) ss[i] = rd_str(buf, pos, map_size);
        out->v.arr.data = ss;
        return;
    }
    size_t esz = scalar_size(atype);
    if (esz == 0) die("unsupported GGUF array type %d", atype);
    unsigned char *data = xmalloc((size_t)n * esz);
    for (uint64_t i = 0; i < n; i++) {
        GGUFValue tmp;
        read_value(buf, pos, map_size, atype, &tmp);
        memcpy(data + i * esz, &tmp.v, esz);
    }
    out->v.arr.data = data;
}

static void read_value(const unsigned char *buf, size_t *pos, size_t map_size, int vtype, GGUFValue *out) {
    memset(out, 0, sizeof(*out));
    out->type = vtype;
    switch (vtype) {
    case GGUF_U8:
        out->v.u = rd_u8(buf, pos);
        break;
    case GGUF_I8:
        out->v.i = rd_i8(buf, pos);
        break;
    case GGUF_U16:
        out->v.u = rd_u16(buf, pos);
        break;
    case GGUF_I16:
        out->v.i = rd_i16(buf, pos);
        break;
    case GGUF_U32:
        out->v.u = rd_u32(buf, pos);
        break;
    case GGUF_I32:
        out->v.i = rd_i32(buf, pos);
        break;
    case GGUF_F32:
        out->v.d = rd_f32(buf, pos);
        break;
    case GGUF_BOOL:
        out->v.b = rd_u8(buf, pos) != 0;
        break;
    case GGUF_STR:
        out->v.s = rd_str(buf, pos, map_size);
        break;
    case GGUF_U64:
        out->v.u = rd_u64(buf, pos);
        break;
    case GGUF_I64:
        out->v.i = rd_i64(buf, pos);
        break;
    case GGUF_F64:
        out->v.d = rd_f64(buf, pos);
        break;
    case GGUF_ARR:
        read_array(buf, pos, map_size, out);
        break;
    default:
        die("unknown GGUF type %d", vtype);
    }
}

static void gguf_value_free(GGUFValue *v) {
    if (v->type == GGUF_STR) {
        free(v->v.s);
        v->v.s = NULL;
    } else if (v->type == GGUF_ARR) {
        if (v->v.arr.type == GGUF_STR) {
            char **ss = (char **)v->v.arr.data;
            for (int i = 0; i < v->v.arr.n; i++) free(ss[i]);
        }
        free(v->v.arr.data);
        v->v.arr.data = NULL;
    }
}

GGUFFile *open_gguf(const char *path) {
    GGUFFile *f = xcalloc(1, sizeof(GGUFFile));
    f->path = xstrdup(path);
    f->fd = -1;
    f->fd = open(path, O_RDONLY);
    if (f->fd < 0) die("cannot open %s: %s", path, strerror(errno));
    struct stat st;
    if (fstat(f->fd, &st) != 0) die("fstat %s failed", path);
    f->map_size = (size_t)st.st_size;
    f->map = mmap(NULL, f->map_size, PROT_READ, MAP_PRIVATE, f->fd, 0);
    if (f->map == MAP_FAILED) die("mmap %s failed: %s", path, strerror(errno));

    const unsigned char *buf = (const unsigned char *)f->map;
    if (f->map_size < 24 || memcmp(buf, GGUF_MAGIC, 4) != 0) die("%s is not a GGUF file", path);
    size_t pos = 4;
    f->version = (int)rd_u32(buf, &pos);
    if (f->version != 3) die("only GGUF v3 is supported, got %d", f->version);
    uint64_t n_tensors = rd_u64(buf, &pos);
    uint64_t n_kv = rd_u64(buf, &pos);

    f->n_kv = (int)n_kv;
    f->kv_keys = xcalloc((size_t)n_kv, sizeof(char *));
    f->kv_vals = xcalloc((size_t)n_kv, sizeof(GGUFValue));
    for (uint64_t i = 0; i < n_kv; i++) {
        f->kv_keys[i] = rd_str(buf, &pos, f->map_size);
        int vtype = (int)rd_u32(buf, &pos);
        read_value(buf, &pos, f->map_size, vtype, &f->kv_vals[i]);
    }

    f->n_tensors = (int)n_tensors;
    f->tensors = xcalloc((size_t)n_tensors, sizeof(TensorInfo));
    for (uint64_t i = 0; i < n_tensors; i++) {
        TensorInfo *t = &f->tensors[i];
        t->name = rd_str(buf, &pos, f->map_size);
        t->n_dims = (int)rd_u32(buf, &pos);
        if (t->n_dims > 8) die("too many dims for %s", t->name);
        for (int d = 0; d < t->n_dims; d++) t->dims[d] = rd_u64(buf, &pos);
        t->ggml_type = (int)rd_u32(buf, &pos);
        t->offset = rd_u64(buf, &pos);
    }

    int alignment = GGUF_DEFAULT_ALIGNMENT;
    const GGUFValue *al = gguf_get(f, "general.alignment");
    if (al) alignment = (int)al->v.u;
    f->data_offset = (pos + (size_t)alignment - 1) / (size_t)alignment * (size_t)alignment;
    return f;
}

void gguf_unmap(GGUFFile *f) {
    if (!f) return;
    if (f->map && f->map != MAP_FAILED) munmap(f->map, f->map_size);
    f->map = NULL;
    if (f->fd >= 0) close(f->fd);
    f->fd = -1;
}

void gguf_close(GGUFFile *f) {
    if (!f) return;
    gguf_unmap(f);
    for (int i = 0; i < f->n_kv; i++) {
        free(f->kv_keys[i]);
        gguf_value_free(&f->kv_vals[i]);
    }
    free(f->kv_keys);
    free(f->kv_vals);
    for (int i = 0; i < f->n_tensors; i++) free(f->tensors[i].name);
    free(f->tensors);
    free(f->path);
    free(f);
}

const GGUFValue *gguf_get(const GGUFFile *f, const char *key) {
    for (int i = 0; i < f->n_kv; i++) {
        if (strcmp(f->kv_keys[i], key) == 0) return &f->kv_vals[i];
    }
    return NULL;
}

const TensorInfo *gguf_find_tensor(const GGUFFile *f, const char *name) {
    for (int i = 0; i < f->n_tensors; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) return &f->tensors[i];
    }
    return NULL;
}

static int kv_int(const GGUFFile *f, const char *key, int def, int required) {
    const GGUFValue *v = gguf_get(f, key);
    if (!v) {
        if (required) die("missing GGUF key %s", key);
        return def;
    }
    if (v->type == GGUF_BOOL) return v->v.b;
    if (v->type == GGUF_F32 || v->type == GGUF_F64) return (int)v->v.d;
    if (v->type == GGUF_I8 || v->type == GGUF_I16 || v->type == GGUF_I32 || v->type == GGUF_I64) return (int)v->v.i;
    return (int)v->v.u;
}

static float kv_float(const GGUFFile *f, const char *key, float def, int required) {
    const GGUFValue *v = gguf_get(f, key);
    if (!v) {
        if (required) die("missing GGUF key %s", key);
        return def;
    }
    if (v->type == GGUF_F32 || v->type == GGUF_F64) return (float)v->v.d;
    return (float)v->v.u;
}

static const char *kv_str(const GGUFFile *f, const char *key, const char *def) {
    const GGUFValue *v = gguf_get(f, key);
    if (!v || v->type != GGUF_STR) return def;
    return v->v.s;
}

static void kv_key(char *buf, size_t n, const char *arch, const char *suffix) {
    snprintf(buf, n, "%s.%s", arch, suffix);
}

static void read_rope_sections(const GGUFFile *f, const char *key, int *out) {
    out[0] = out[1] = out[2] = out[3] = 0;
    const GGUFValue *v = gguf_get(f, key);
    if (!v || v->type != GGUF_ARR) return;
    int n = v->v.arr.n < 4 ? v->v.arr.n : 4;
    const unsigned char *p = (const unsigned char *)v->v.arr.data;
    int t = v->v.arr.type;
    for (int i = 0; i < n; i++) {
        if (t == GGUF_I32 || t == GGUF_U32) {
            int32_t x;
            memcpy(&x, p + (size_t)i * 4, 4);
            out[i] = (int)x;
        } else if (t == GGUF_I64 || t == GGUF_U64) {
            int64_t x;
            memcpy(&x, p + (size_t)i * 8, 8);
            out[i] = (int)x;
        }
    }
}

int hparams_from_kv(const GGUFFile *f, LlamaHParams *hp) {
    memset(hp, 0, sizeof(*hp));
    const char *arch = kv_str(f, "general.architecture", "");
    const char *pfx;
    if (strcmp(arch, "llama") == 0) {
        hp->arch = LLM_ARCH_LLAMA;
        pfx = "llama";
    } else if (strcmp(arch, "qwen2") == 0) {
        hp->arch = LLM_ARCH_QWEN2;
        pfx = "qwen2";
        hp->rope_neox = 1;
    } else if (strcmp(arch, "qwen35") == 0) {
        hp->arch = LLM_ARCH_QWEN35;
        pfx = "qwen35";
        hp->rope_neox = 1;
    } else {
        die("unsupported architecture %s", arch);
        return -1;
    }
    char key[128];
    kv_key(key, sizeof(key), pfx, "embedding_length");
    hp->n_embd = kv_int(f, key, 0, 1);
    kv_key(key, sizeof(key), pfx, "attention.head_count");
    hp->n_head = kv_int(f, key, 0, 1);
    kv_key(key, sizeof(key), pfx, "attention.key_length");
    hp->head_dim = kv_int(f, key, hp->n_embd / hp->n_head, 0);
    kv_key(key, sizeof(key), pfx, "block_count");
    hp->n_layer = kv_int(f, key, 0, 1);
    kv_key(key, sizeof(key), pfx, "feed_forward_length");
    hp->n_ff = kv_int(f, key, 0, 1);
    kv_key(key, sizeof(key), pfx, "attention.head_count_kv");
    hp->n_head_kv = kv_int(f, key, 0, 1);
    kv_key(key, sizeof(key), pfx, "rope.dimension_count");
    hp->n_rot = kv_int(f, key, hp->head_dim, 0);
    kv_key(key, sizeof(key), pfx, "vocab_size");
    hp->n_vocab = kv_int(f, key, 0, 0);
    if (hp->n_vocab <= 0) {
        const GGUFValue *toks = gguf_get(f, "tokenizer.ggml.tokens");
        if (toks && toks->type == GGUF_ARR) hp->n_vocab = toks->v.arr.n;
    }
    kv_key(key, sizeof(key), pfx, "context_length");
    hp->n_ctx = kv_int(f, key, 0, 1);
    kv_key(key, sizeof(key), pfx, "attention.layer_norm_rms_epsilon");
    hp->rms_eps = kv_float(f, key, 1e-5f, 1);
    kv_key(key, sizeof(key), pfx, "rope.freq_base");
    hp->rope_theta = kv_float(f, key, 10000.0f, 1);
    kv_key(key, sizeof(key), pfx, "rope.dimension_sections");
    read_rope_sections(f, key, hp->rope_sections);
    kv_key(key, sizeof(key), pfx, "full_attention_interval");
    hp->full_attention_interval = kv_int(f, key, 4, 0);
    kv_key(key, sizeof(key), pfx, "nextn_predict_layers");
    hp->n_layer_nextn = kv_int(f, key, 0, 0);
    if (hp->n_layer_nextn < 0) hp->n_layer_nextn = 0;
    if (hp->n_layer_nextn >= hp->n_layer) die("nextn_predict_layers >= block_count");
    hp->n_layer_fwd = hp->n_layer - hp->n_layer_nextn;
    kv_key(key, sizeof(key), pfx, "ssm.conv_kernel");
    hp->ssm_d_conv = kv_int(f, key, 0, 0);
    kv_key(key, sizeof(key), pfx, "ssm.inner_size");
    hp->ssm_d_inner = kv_int(f, key, 0, 0);
    kv_key(key, sizeof(key), pfx, "ssm.state_size");
    hp->ssm_d_state = kv_int(f, key, 0, 0);
    kv_key(key, sizeof(key), pfx, "ssm.time_step_rank");
    hp->ssm_dt_rank = kv_int(f, key, 0, 0);
    kv_key(key, sizeof(key), pfx, "ssm.group_count");
    hp->ssm_n_group = kv_int(f, key, 0, 0);
    /* Qwen3.5-9B / MiniCPM: thinking off. GGUF jinja turns it on when the kwarg
       is omitted, which streams a long CoT (and often a repetition loop). */
    hp->enable_thinking = 0;
    hp->bos_id = kv_int(f, "tokenizer.ggml.bos_token_id", 1, 0);
    hp->eos_id = kv_int(f, "tokenizer.ggml.eos_token_id", 2, 0);
    hp->unk_id = kv_int(f, "tokenizer.ggml.unknown_token_id", 0, 0);
    hp->pad_id = kv_int(f, "tokenizer.ggml.padding_token_id", 2, 0);
    hp->add_space_prefix = kv_int(f, "tokenizer.ggml.add_space_prefix", 0, 0);
    hp->chat_template = xstrdup(kv_str(f, "tokenizer.chat_template", ""));
    hp->tokenizer_pre = xstrdup(kv_str(f, "tokenizer.ggml.pre", ""));
    hp->model_name = xstrdup(kv_str(f, "general.name", ""));
    return 0;
}

void hparams_free(LlamaHParams *hp) {
    free(hp->chat_template);
    free(hp->tokenizer_pre);
    free(hp->model_name);
    hp->chat_template = NULL;
    hp->tokenizer_pre = NULL;
    hp->model_name = NULL;
}

static uint64_t n_elements_u64(const TensorInfo *info) {
    uint64_t n = 1;
    for (int i = 0; i < info->n_dims; i++) n *= info->dims[i];
    return n;
}

static int n_elements_dims(const TensorInfo *info) {
    uint64_t n = n_elements_u64(info);
    if (n > (uint64_t)INT_MAX) die("tensor %s is too large", info->name);
    return (int)n;
}

size_t gguf_dequant_f32_bytes(const GGUFFile *f) {
    size_t bytes = 0;
    for (int i = 0; i < f->n_tensors; i++) {
        bytes += (size_t)n_elements_u64(&f->tensors[i]) * sizeof(float);
    }
    return bytes;
}

size_t gguf_packed_bytes(const GGUFFile *f) {
    size_t bytes = 0;
    for (int i = 0; i < f->n_tensors; i++) {
        const TensorInfo *info = &f->tensors[i];
        bytes += ggml_nbytes(info->ggml_type, n_elements_dims(info));
    }
    return bytes;
}

static const void *tensor_blob(const GGUFFile *gguf, const TensorInfo *info) {
    return (const unsigned char *)gguf->map + gguf->data_offset + (size_t)info->offset;
}

static int row_keep_quantized(const TensorInfo *info) {
    if (info->ggml_type == GGML_F32) return 0;
    if (info->n_dims < 2) return 0;
    int inner = (int)info->dims[0];
    int qk = ggml_blck_size(info->ggml_type);
    return qk > 0 && inner % qk == 0;
}

static float *dequant_tensor(const GGUFFile *gguf, const TensorInfo *info) {
    size_t start = gguf->data_offset + (size_t)info->offset;
    int n = n_elements_dims(info);
    const unsigned char *blob = (const unsigned char *)gguf->map + start;
    float *flat = xmalloc((size_t)n * sizeof(float));
    if (info->ggml_type == GGML_F32) {
        if (dequantize_f32(blob, n, flat) != 0) die("f32 dequant failed for %s", info->name);
    } else if (info->ggml_type == GGML_Q8_0) {
        if (dequantize_q8_0(blob, n, flat) != 0) die("Q8_0 dequant failed for %s", info->name);
    } else if (info->ggml_type == GGML_Q4_K) {
        if (dequantize_q4_k(blob, n, flat) != 0) die("Q4_K dequant failed for %s", info->name);
    } else if (info->ggml_type == GGML_Q6_K) {
        if (dequantize_q6_k(blob, n, flat) != 0) die("Q6_K dequant failed for %s", info->name);
    } else if (info->ggml_type == GGML_IQ4_XS) {
        if (dequantize_iq4_xs(blob, n, flat) != 0) die("IQ4_XS dequant failed for %s", info->name);
    } else {
        die("unsupported ggml type %d for %s", info->ggml_type, info->name);
    }
    return flat;
}

LoadedModel *load_model(const char *path, int load_weights, int progress) {
    LoadedModel *m = xcalloc(1, sizeof(LoadedModel));
    m->gguf = open_gguf(path);
    hparams_from_kv(m->gguf, &m->hparams);
    if (!load_weights) return m;
    if (progress) {
        size_t packed = gguf_packed_bytes(m->gguf);
        size_t as_f32 = gguf_dequant_f32_bytes(m->gguf);
        printf("  mmap quantized weights (~%.2f GiB packed, f32 would be ~%.2f GiB)\n",
               (double)packed / (1024.0 * 1024.0 * 1024.0),
               (double)as_f32 / (1024.0 * 1024.0 * 1024.0));
        fflush(stdout);
    }
    m->n_weights = m->gguf->n_tensors;
    m->weights = xcalloc((size_t)m->n_weights, sizeof(WeightTensor));
    int n_dequant = 0;
    for (int i = 0; i < m->n_weights; i++) {
        const TensorInfo *info = &m->gguf->tensors[i];
        WeightTensor *w = &m->weights[i];
        w->name = xstrdup(info->name);
        w->ndim = info->n_dims;
        w->n_elements = n_elements_dims(info);
        w->ggml_type = info->ggml_type;
        for (int d = 0; d < info->n_dims; d++) w->shape[d] = (int)info->dims[info->n_dims - 1 - d];
        if (info->ggml_type == GGML_F32) {
            w->data = tensor_blob(m->gguf, info);
            w->nbytes = (size_t)w->n_elements * sizeof(float);
            w->owned = 0;
        } else if (row_keep_quantized(info)) {
            w->data = tensor_blob(m->gguf, info);
            w->nbytes = ggml_nbytes(info->ggml_type, w->n_elements);
            w->owned = 0;
        } else {
            w->data = dequant_tensor(m->gguf, info);
            w->ggml_type = GGML_F32;
            w->nbytes = (size_t)w->n_elements * sizeof(float);
            w->owned = 1;
            n_dequant++;
        }
        if (progress && (i + 1 == m->n_weights || (i + 1) % 40 == 0)) {
            printf("  load %d/%d tensors (%d small as f32)\n", i + 1, m->n_weights, n_dequant);
            fflush(stdout);
        }
    }
    return m;
}

void loaded_model_free(LoadedModel *m) {
    if (!m) return;
    for (int i = 0; i < m->n_weights; i++) {
        free(m->weights[i].name);
        if (m->weights[i].owned) free((void *)m->weights[i].data);
    }
    free(m->weights);
    hparams_free(&m->hparams);
    gguf_close(m->gguf);
    free(m);
}

const WeightTensor *loaded_find_weight(const LoadedModel *m, const char *name) {
    for (int i = 0; i < m->n_weights; i++) {
        if (strcmp(m->weights[i].name, name) == 0) return &m->weights[i];
    }
    return NULL;
}
