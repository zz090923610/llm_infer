#ifndef LLM_HASHMAP_H
#define LLM_HASHMAP_H

typedef struct {
    char *key;
    int value;
    int state; /* 0 empty, 1 used */
} HashSlot;

typedef struct {
    HashSlot *slots;
    int cap;
    int size;
} HashMap;

void hashmap_init(HashMap *m);
void hashmap_free(HashMap *m);
void hashmap_put(HashMap *m, const char *key, int value);
/* returns 1 if found */
int hashmap_get(HashMap *m, const char *key, int *out);

#endif
