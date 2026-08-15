#ifndef LLM_GGUF_H
#define LLM_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define GGUF_MAGIC "GGUF"
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGML_F32 0
#define GGML_Q8_0 8

enum {
    GGUF_U8 = 0,
    GGUF_I8,
    GGUF_U16,
    GGUF_I16,
    GGUF_U32,
    GGUF_I32,
    GGUF_F32,
    GGUF_BOOL,
    GGUF_STR,
    GGUF_ARR,
    GGUF_U64,
    GGUF_I64,
    GGUF_F64
};

typedef struct {
    int type;
    union {
        uint64_t u;
        int64_t i;
        double d;
        int b;
        char *s;
        struct {
            int type;
            int n;
            /* strings: char**; scalars: packed array of the element type */
            void *data;
        } arr;
    } v;
} GGUFValue;

typedef struct {
    char *name;
    int n_dims;
    uint64_t dims[8]; /* ggml order, ne[0] first */
    int ggml_type;
    uint64_t offset;
} TensorInfo;

typedef struct {
    char *path;
    int version;
    int n_kv;
    char **kv_keys;
    GGUFValue *kv_vals;
    int n_tensors;
    TensorInfo *tensors;
    size_t data_offset;
    void *map;
    size_t map_size;
    int fd;
} GGUFFile;

typedef struct {
    int n_layer;
    int n_embd;
    int n_ff;
    int n_head;
    int n_head_kv;
    int n_rot;
    int head_dim;
    int n_vocab;
    int n_ctx;
    float rms_eps;
    float rope_theta;
    int bos_id;
    int eos_id;
    int unk_id;
    int pad_id;
    int add_space_prefix;
    char *chat_template;
    char *tokenizer_pre;
} LlamaHParams;

typedef struct {
    char *name;
    float *data;
    int ndim;
    int shape[8]; /* NumPy / row-major order (reversed ggml) */
    int n_elements;
} WeightTensor;

typedef struct {
    GGUFFile *gguf;
    LlamaHParams hparams;
    int n_weights;
    WeightTensor *weights;
} LoadedModel;

GGUFFile *open_gguf(const char *path);
void gguf_close(GGUFFile *f);
const GGUFValue *gguf_get(const GGUFFile *f, const char *key);
const TensorInfo *gguf_find_tensor(const GGUFFile *f, const char *name);
/* Bytes needed to hold every tensor as float32 after dequant. */
size_t gguf_dequant_f32_bytes(const GGUFFile *f);

int hparams_from_kv(const GGUFFile *f, LlamaHParams *hp);
void hparams_free(LlamaHParams *hp);

LoadedModel *load_model(const char *path, int dequant, int progress);
void loaded_model_free(LoadedModel *m);
const WeightTensor *loaded_find_weight(const LoadedModel *m, const char *name);

#endif
