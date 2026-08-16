#ifndef LLM_TOKENIZER_H
#define LLM_TOKENIZER_H

#include "gguf.h"
#include "hashmap.h"
#include "util.h"

typedef struct {
    char *role;
    char *content;
} ChatMessage;

typedef struct {
    char **vocab;
    int n_vocab;
    HashMap token_to_id;
    HashMap merges_rank;
    char **specials;
    int n_specials;
    HashMap special_set;
    LlamaHParams hparams;
    int *stop_ids;
    int n_stop;
} Tokenizer;

typedef struct {
    Tokenizer *tok;
    int skip_special;
    int in_think;
    ByteVec buf;
} StreamDecoder;

char *apply_chat_template(const Tokenizer *tok, const ChatMessage *msgs, int n, int add_generation_prompt);

Tokenizer *tokenizer_from_gguf(const GGUFFile *gguf, const LlamaHParams *hp);
Tokenizer *tokenizer_from_file(const char *path);
void tokenizer_free(Tokenizer *t);

int tokenizer_encode(Tokenizer *t, const char *text, int parse_special, IntVec *out);
char *tokenizer_decode(Tokenizer *t, const int *ids, int n, int skip_special);

void stream_decoder_init(StreamDecoder *d, Tokenizer *t, int skip_special);
void stream_decoder_free(StreamDecoder *d);
/* Returns a newly allocated string (possibly empty); caller frees. */
char *stream_decoder_push(StreamDecoder *d, int token_id);
char *stream_decoder_flush(StreamDecoder *d);

#endif
