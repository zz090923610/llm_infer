#ifndef LLM_HASHMAP_H
#define LLM_HASHMAP_H

#include <stdint.h>

typedef struct {
    char *key;
    uint32_t hash;
    int value;
    int state; /* 0 empty, 1 used */
} HashSlot;

typedef struct {
    HashSlot *slots;
    int cap;
    int size;
    int own_keys; /* 1 = strdup/free keys; 0 = borrowed pointers */
} HashMap;

typedef struct {
    uint64_t key;
    int value;
    int state;
} HashSlotU64;

typedef struct {
    HashSlotU64 *slots;
    int cap;
    int size;
} HashMapU64;

void hashmap_init(HashMap *m);
void hashmap_init_sized(HashMap *m, int expected);
void hashmap_init_ref(HashMap *m, int expected);
void hashmap_free(HashMap *m);
void hashmap_put(HashMap *m, const char *key, int value);
int hashmap_get(HashMap *m, const char *key, int *out);

void hashmap_u64_init_sized(HashMapU64 *m, int expected);
void hashmap_u64_free(HashMapU64 *m);
void hashmap_u64_put(HashMapU64 *m, uint64_t key, int value);
int hashmap_u64_get(HashMapU64 *m, uint64_t key, int *out);

/* Chained string map: 8K bucket heads stay in L1; nodes are dense. */
typedef struct {
    uint32_t hash;
    int value;
    int next;
    const char *key;
} StrMapNode;

typedef struct {
    int *head;
    StrMapNode *nodes;
    int nbuckets;
    int n;
    int cap;
} StrMap;

void strmap_init(StrMap *m, int expected);
void strmap_free(StrMap *m);
void strmap_put(StrMap *m, const char *key, int value);
int strmap_get(StrMap *m, const char *key, int *out);

#endif
