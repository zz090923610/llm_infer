#ifndef LLM_HEAP_H
#define LLM_HEAP_H

/* Min-heap of (rank, left, right) matching Python heapq tuple order. */

typedef struct {
    int rank;
    int left;
    int right;
} HeapItem;

typedef struct {
    HeapItem *data;
    int n;
    int cap;
} MinHeap;

void heap_init(MinHeap *h);
void heap_free(MinHeap *h);
void heap_clear(MinHeap *h);
void heap_push(MinHeap *h, int rank, int left, int right);
int heap_pop(MinHeap *h, HeapItem *out); /* 1 if popped */

#endif
