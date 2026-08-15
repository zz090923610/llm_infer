#include "hashmap.h"
#include "util.h"

static uint32_t fnv1a(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (unsigned char)*s;
        h *= 16777619u;
    }
    return h;
}

void hashmap_init(HashMap *m) {
    m->cap = 64;
    m->size = 0;
    m->slots = xcalloc((size_t)m->cap, sizeof(HashSlot));
}

void hashmap_free(HashMap *m) {
    if (!m->slots) return;
    for (int i = 0; i < m->cap; i++) free(m->slots[i].key);
    free(m->slots);
    m->slots = NULL;
    m->cap = m->size = 0;
}

static void hashmap_grow(HashMap *m) {
    HashSlot *old = m->slots;
    int old_cap = m->cap;
    m->cap *= 2;
    m->size = 0;
    m->slots = xcalloc((size_t)m->cap, sizeof(HashSlot));
    for (int i = 0; i < old_cap; i++) {
        if (old[i].state) {
            hashmap_put(m, old[i].key, old[i].value);
            free(old[i].key);
        }
    }
    free(old);
}

void hashmap_put(HashMap *m, const char *key, int value) {
    if ((m->size + 1) * 10 >= m->cap * 7) hashmap_grow(m);
    uint32_t h = fnv1a(key);
    for (int i = 0; i < m->cap; i++) {
        int idx = (int)((h + (uint32_t)i) & (uint32_t)(m->cap - 1));
        HashSlot *s = &m->slots[idx];
        if (!s->state) {
            s->key = xstrdup(key);
            s->value = value;
            s->state = 1;
            m->size++;
            return;
        }
        if (strcmp(s->key, key) == 0) {
            s->value = value;
            return;
        }
    }
    die("hashmap full");
}

int hashmap_get(HashMap *m, const char *key, int *out) {
    if (!m->slots) return 0;
    uint32_t h = fnv1a(key);
    for (int i = 0; i < m->cap; i++) {
        int idx = (int)((h + (uint32_t)i) & (uint32_t)(m->cap - 1));
        HashSlot *s = &m->slots[idx];
        if (!s->state) return 0;
        if (strcmp(s->key, key) == 0) {
            if (out) *out = s->value;
            return 1;
        }
    }
    return 0;
}
