#include "pim_func/hybrid.h"

#include <stdio.h>
#include <stdlib.h>

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    printf("pim_hybrid_stub start V0=%#llx\n", (unsigned long long)pim_hybrid_v0());
    return pim_hybrid_guest_loop();
}
