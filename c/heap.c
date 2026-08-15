#include "heap.h"
#include "util.h"

static int heap_less(HeapItem a, HeapItem b) {
    if (a.rank != b.rank) return a.rank < b.rank;
    if (a.left != b.left) return a.left < b.left;
    return a.right < b.right;
}

void heap_init(MinHeap *h) {
    h->data = NULL;
    h->n = 0;
    h->cap = 0;
}

void heap_free(MinHeap *h) {
    free(h->data);
    h->data = NULL;
    h->n = h->cap = 0;
}

void heap_clear(MinHeap *h) { h->n = 0; }

void heap_push(MinHeap *h, int rank, int left, int right) {
    if (h->n >= h->cap) {
        h->cap = h->cap ? h->cap * 2 : 32;
        h->data = xrealloc(h->data, (size_t)h->cap * sizeof(HeapItem));
    }
    HeapItem item = {rank, left, right};
    int i = h->n++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (!heap_less(item, h->data[p])) break;
        h->data[i] = h->data[p];
        i = p;
    }
    h->data[i] = item;
}

int heap_pop(MinHeap *h, HeapItem *out) {
    if (h->n == 0) return 0;
    *out = h->data[0];
    HeapItem last = h->data[--h->n];
    if (h->n == 0) return 1;
    int i = 0;
    for (;;) {
        int l = 2 * i + 1;
        int r = 2 * i + 2;
        int smallest = i;
        HeapItem cur = last;
        if (l < h->n && heap_less(h->data[l], cur)) {
            smallest = l;
            cur = h->data[l];
        }
        if (r < h->n && heap_less(h->data[r], cur)) {
            smallest = r;
        }
        if (smallest == i) {
            h->data[i] = last;
            break;
        }
        h->data[i] = h->data[smallest];
        i = smallest;
    }
    return 1;
}
