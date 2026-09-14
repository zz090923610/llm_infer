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

static int cap_for(int expected) {
    int cap = 64;
    if (expected < 0) expected = 0;
    /* Load factor 1/2: fewer linear clusters than 0.7. */
    while ((expected + 1) * 2 >= cap) {
        int next = cap * 2;
        if (next < cap) die("hashmap cap overflow");
        cap = next;
    }
    return cap;
}

static uint32_t probe_idx(uint32_t h, uint32_t i, int cap) {
    /* Quadratic probing: i(i+1)/2. Visits all slots when cap is a power of 2. */
    return (h + (i * (i + 1u)) / 2u) & (uint32_t)(cap - 1);
}

void hashmap_init_sized(HashMap *m, int expected) {
    m->cap = cap_for(expected);
    m->size = 0;
    m->own_keys = 1;
    m->slots = xcalloc((size_t)m->cap, sizeof(HashSlot));
}

void hashmap_init_ref(HashMap *m, int expected) {
    hashmap_init_sized(m, expected);
    m->own_keys = 0;
}

void hashmap_init(HashMap *m) {
    hashmap_init_sized(m, 0);
}

void hashmap_free(HashMap *m) {
    if (!m->slots) return;
    if (m->own_keys) {
        for (int i = 0; i < m->cap; i++) free(m->slots[i].key);
    }
    free(m->slots);
    m->slots = NULL;
    m->cap = m->size = 0;
}

static void hashmap_grow(HashMap *m) {
    HashSlot *old = m->slots;
    int old_cap = m->cap;
    m->cap *= 2;
    LLM_STEP("hashmap_grow %d -> %d (size=%d)\n", old_cap, m->cap, m->size);
    m->size = 0;
    m->slots = xcalloc((size_t)m->cap, sizeof(HashSlot));
    for (int i = 0; i < old_cap; i++) {
        if (old[i].state) {
            hashmap_put(m, old[i].key, old[i].value);
            if (m->own_keys) free(old[i].key);
        }
    }
    free(old);
    LLM_STEP("hashmap_grow done cap=%d size=%d\n", m->cap, m->size);
}

void hashmap_put(HashMap *m, const char *key, int value) {
    if ((m->size + 1) * 2 >= m->cap) hashmap_grow(m);
    uint32_t h = fnv1a(key);
    for (uint32_t i = 0; i < (uint32_t)m->cap; i++) {
        HashSlot *s = &m->slots[probe_idx(h, i, m->cap)];
        if (!s->state) {
            s->key = m->own_keys ? xstrdup(key) : (char *)key;
            s->hash = h;
            s->value = value;
            s->state = 1;
            m->size++;
            return;
        }
        if (s->hash == h && strcmp(s->key, key) == 0) {
            s->value = value;
            return;
        }
    }
    die("hashmap full");
}

int hashmap_get(HashMap *m, const char *key, int *out) {
    if (!m->slots) return 0;
    uint32_t h = fnv1a(key);
    for (uint32_t i = 0; i < (uint32_t)m->cap; i++) {
        HashSlot *s = &m->slots[probe_idx(h, i, m->cap)];
        if (!s->state) return 0;
        if (s->hash == h && strcmp(s->key, key) == 0) {
            if (out) *out = s->value;
            return 1;
        }
    }
    return 0;
}

static uint32_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (uint32_t)x;
}

void hashmap_u64_init_sized(HashMapU64 *m, int expected) {
    m->cap = cap_for(expected);
    m->size = 0;
    m->slots = xcalloc((size_t)m->cap, sizeof(HashSlotU64));
}

void hashmap_u64_free(HashMapU64 *m) {
    free(m->slots);
    m->slots = NULL;
    m->cap = m->size = 0;
}

static void hashmap_u64_grow(HashMapU64 *m) {
    HashSlotU64 *old = m->slots;
    int old_cap = m->cap;
    m->cap *= 2;
    m->size = 0;
    m->slots = xcalloc((size_t)m->cap, sizeof(HashSlotU64));
    for (int i = 0; i < old_cap; i++) {
        if (old[i].state) hashmap_u64_put(m, old[i].key, old[i].value);
    }
    free(old);
}

void hashmap_u64_put(HashMapU64 *m, uint64_t key, int value) {
    if ((m->size + 1) * 2 >= m->cap) hashmap_u64_grow(m);
    uint32_t h = mix64(key);
    for (uint32_t i = 0; i < (uint32_t)m->cap; i++) {
        HashSlotU64 *s = &m->slots[probe_idx(h, i, m->cap)];
        if (!s->state) {
            s->key = key;
            s->value = value;
            s->state = 1;
            m->size++;
            return;
        }
        if (s->key == key) {
            s->value = value;
            return;
        }
    }
    die("hashmap_u64 full");
}

int hashmap_u64_get(HashMapU64 *m, uint64_t key, int *out) {
    if (!m->slots) return 0;
    uint32_t h = mix64(key);
    for (uint32_t i = 0; i < (uint32_t)m->cap; i++) {
        HashSlotU64 *s = &m->slots[probe_idx(h, i, m->cap)];
        if (!s->state) return 0;
        if (s->key == key) {
            if (out) *out = s->value;
            return 1;
        }
    }
    return 0;
}

void strmap_init(StrMap *m, int expected) {
    int nb = 1024;
    if (expected < 0) expected = 0;
    while (nb < 8192 && nb < expected / 4) nb *= 2;
    m->nbuckets = nb;
    m->n = 0;
    m->cap = expected > 0 ? expected : 16;
    m->head = xmalloc((size_t)nb * sizeof(int));
    for (int i = 0; i < nb; i++) m->head[i] = -1;
    m->nodes = xmalloc((size_t)m->cap * sizeof(StrMapNode));
}

void strmap_free(StrMap *m) {
    free(m->head);
    free(m->nodes);
    m->head = NULL;
    m->nodes = NULL;
    m->n = m->cap = m->nbuckets = 0;
}

void strmap_put(StrMap *m, const char *key, int value) {
    if (m->n >= m->cap) {
        m->cap = m->cap ? m->cap * 2 : 16;
        m->nodes = xrealloc(m->nodes, (size_t)m->cap * sizeof(StrMapNode));
    }
    uint32_t h = fnv1a(key);
    int b = (int)(h & (uint32_t)(m->nbuckets - 1));
    StrMapNode *node = &m->nodes[m->n];
    node->hash = h;
    node->value = value;
    node->key = key;
    node->next = m->head[b];
    m->head[b] = m->n;
    m->n++;
}

int strmap_get(StrMap *m, const char *key, int *out) {
    if (!m->head) return 0;
    uint32_t h = fnv1a(key);
    int b = (int)(h & (uint32_t)(m->nbuckets - 1));
    for (int i = m->head[b]; i >= 0; i = m->nodes[i].next) {
        if (m->nodes[i].hash == h && strcmp(m->nodes[i].key, key) == 0) {
            if (out) *out = m->nodes[i].value;
            return 1;
        }
    }
    return 0;
}
