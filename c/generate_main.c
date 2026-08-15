#include "generate.h"
#include "gguf.h"
#include "util.h"

#ifndef LLM_DEFAULT_MODEL
#define LLM_DEFAULT_MODEL "../nanogpt-chat-q8_0.gguf"
#endif

static void usage(const char *argv0) {
    fprintf(stderr,
            "Usage: %s [--model PATH] [--prompt TEXT] [--max-tokens N] [--temp F] [--top-p F] "
            "[--top-k N] [--batch-demo]\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *model_path = LLM_DEFAULT_MODEL;
    const char *prompt = "Hello";
    int max_tokens = 64;
    float temp = 0.8f;
    float top_p = 0.9f;
    int top_k = 0;
    int batch_demo = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) max_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) top_p = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-demo") == 0) batch_demo = 1;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else {
            usage(argv[0]);
            return 1;
        }
    }

    printf("loading %s\n", model_path);
    fflush(stdout);
    LoadedModel *loaded = load_model(model_path, 1, 1);
    LlamaModel *model = llama_model_init(loaded);
    Tokenizer *tok = tokenizer_from_gguf(loaded->gguf, &loaded->hparams);

    if (batch_demo) {
        ChatMessage m0 = {"user", "Say hi in one word."};
        ChatMessage m1 = {"user", "2+2="};
        char *p0 = apply_chat_template(&m0, 1, 1);
        char *p1 = apply_chat_template(&m1, 1, 1);
        char *prompts[2] = {p0, p1};
        printf("batched greedy decode:\n");
        fflush(stdout);
        IntVec *outs = generate_batch(model, tok, prompts, 2, max_tokens, 0.0f, 0, 1.0f, 1);
        for (int i = 0; i < 2; i++) {
            printf("---\n");
            char *text = tokenizer_decode(tok, outs[i].data, outs[i].n, 1);
            printf("%s\n", text);
            free(text);
            intvec_free(&outs[i]);
        }
        free(outs);
        free(p0);
        free(p1);
    } else {
        ChatMessage msg = {"user", (char *)prompt};
        char *text = apply_chat_template(&msg, 1, 1);
        char *out = generate_text(model, tok, text, max_tokens, temp, top_k, top_p, 1, 1, NULL);
        free(out);
        free(text);
    }

    tokenizer_free(tok);
    llama_model_free(model);
    loaded_model_free(loaded);
    return 0;
}
