#include "pim_func/hybrid.h"

#include "pim_func/device.h"
#include "pim_func/fp16.h"
#include "pim_func/guest_mmio.h"
#include "pim_func/hook.h"
#include "pim_func/isa.h"
#include "pim_func/layout.h"
#include "pim_func/mem.h"
#include "pim_func/pool.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <string>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <unistd.h>

namespace {

constexpr uint64_t kAlign = 64;

PimHybridHeader *g_hdr = nullptr;
void *g_base = nullptr;
uint64_t g_size = 0;
int g_fd = -1;
pthread_t g_mac_th;
int g_mac_started = 0;
int g_server = 0;

std::string shm_name() {
    const char *e = std::getenv("PIM_HYBRID_SHM");
    if (e && e[0]) {
        if (e[0] == '/') return e;
        return std::string("/") + e;
    }
    return "/pim_hybrid";
}

uint64_t env_size(uint64_t fallback) {
    const char *e = std::getenv("PIM_HYBRID_SIZE");
    if (!e || !e[0]) return fallback;
    char *end = nullptr;
    unsigned long long v = std::strtoull(e, &end, 0);
    if (v < sizeof(PimHybridHeader) + 4096) return fallback;
    return static_cast<uint64_t>(v);
}

long futex_wait(uint32_t *addr, uint32_t expect) {
    return syscall(SYS_futex, addr, FUTEX_WAIT, expect, nullptr, nullptr, 0);
}

long futex_wake(uint32_t *addr, int n) {
    return syscall(SYS_futex, addr, FUTEX_WAKE, n, nullptr, nullptr, 0);
}

uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }

int mmap_at_v0(int fd, uint64_t size, int create) {
    void *want = reinterpret_cast<void *>(PIM_HYBRID_V0);
    int flags = MAP_SHARED | MAP_FIXED;
#ifdef MAP_FIXED_NOREPLACE
    flags = MAP_SHARED | MAP_FIXED_NOREPLACE;
#endif
    void *p = mmap(want, static_cast<size_t>(size), PROT_READ | PROT_WRITE, flags, fd, 0);
    if (p == MAP_FAILED) {
        p = mmap(want, static_cast<size_t>(size), PROT_READ | PROT_WRITE,
                 MAP_SHARED | MAP_FIXED, fd, 0);
    }
    if (p == MAP_FAILED) {
        std::fprintf(stderr, "pim_hybrid: mmap V0=%#llx size=%llu failed: %s\n",
                     (unsigned long long)PIM_HYBRID_V0, (unsigned long long)size, std::strerror(errno));
        return -1;
    }
    if (p != want) {
        std::fprintf(stderr, "pim_hybrid: mmap moved to %p, want %#llx\n", p,
                     (unsigned long long)PIM_HYBRID_V0);
        munmap(p, static_cast<size_t>(size));
        return -1;
    }
    g_base = p;
    g_size = size;
    g_hdr = reinterpret_cast<PimHybridHeader *>(p);
    if (create) {
        std::memset(p, 0, sizeof(PimHybridHeader));
        g_hdr->magic = PIM_HYBRID_MAGIC;
        g_hdr->size = size;
        g_hdr->v0 = PIM_HYBRID_V0;
        /* Banks start past the guest-mapped prefix (header + x/y scratch). */
        g_hdr->bump = PIM_HYBRID_GUEST_MAP;
        g_hdr->b_ready = 1;
        std::atomic_thread_fence(std::memory_order_release);
    }
    (void)create;
    return 0;
}

struct MacWaveJob {
    const float *x_f32 = nullptr;
    float *y = nullptr;
    int n_out = 0;
    int n_banks = 0;
    int n_cols = 0;
    int row_base = 0;
    int n_waves = 0;
    int acc = 0;
    int ch = 0;
};

void mac_wave_fn(int tid, int n_threads, void *ctx) {
    auto *j = static_cast<MacWaveJob *>(ctx);
    pim_func::CentGemvAllocation alloc;
    const auto chs = alloc.pim_channels(pim_func::gem5_hw());
    const int n_ch = static_cast<int>(chs.size());
    alignas(32) float accv[pim_func::kBanks];
    for (int wave = tid; wave < j->n_waves; wave += n_threads) {
        for (int ci = 0; ci < n_ch; ci++) {
            pim_func::global_device().mac_wave_into(chs[static_cast<size_t>(ci)],
                                                    j->row_base + wave, j->n_cols, accv, j->x_f32);
            const int o0 = (wave * n_ch + ci) * j->n_banks;
            if (o0 >= j->n_out) break;
            const int n = j->n_out - o0 < j->n_banks ? j->n_out - o0 : j->n_banks;
            if (j->acc) {
                for (int i = 0; i < n; i++) j->y[o0 + i] += accv[i];
            } else {
                std::memcpy(j->y + o0, accv, static_cast<size_t>(n) * sizeof(float));
            }
        }
    }
}

void *mac_worker_main(void *) {
    PimHybridHeader *h = g_hdr;
    uint32_t seen = 0;
    while (h && !h->stop) {
        uint32_t seq = h->gemv_seq;
        if (seq == seen) {
            futex_wait((uint32_t *)&h->gemv_seq, seq);
            continue;
        }
        seen = seq;
        pim_func::hybrid_sync_dram(&pim_func::global_device().dram());
        const float *x = reinterpret_cast<const float *>(static_cast<uintptr_t>(h->x_ptr));
        float *y = reinterpret_cast<float *>(static_cast<uintptr_t>(h->y_ptr));
        if (!x || !y || h->n_out <= 0 || h->n_cols <= 0) {
            h->gemv_done = seq;
            futex_wake((uint32_t *)&h->gemv_done, 1);
            continue;
        }
        MacWaveJob job;
        job.x_f32 = x;
        job.y = y;
        job.n_out = h->n_out;
        job.n_banks = h->n_banks > 0 ? h->n_banks : pim_func::kBanks;
        job.n_cols = h->n_cols;
        job.row_base = h->row_base;
        job.n_waves = h->n_waves > 0 ? h->n_waves : 1;
        job.acc = h->acc;
        job.ch = h->ch;
        int nth = pim_func::pool_n_threads();
        if (nth > job.n_waves) nth = job.n_waves;
        if (nth < 1) nth = 1;
        pim_func::pool_run(mac_wave_fn, &job, nth);
        std::atomic_thread_fence(std::memory_order_release);
        h->gemv_done = seq;
        futex_wake((uint32_t *)&h->gemv_done, 1);
    }
    return nullptr;
}

void mmio_pack_x_from(const float *x, int n_in, int n_cols, int lanes) {
    for (int col = 0; col < n_cols; col++) {
        uint16_t lanes16[16] = {};
        for (int lane = 0; lane < lanes && lane < 16; lane++) {
            const int i = col * lanes + lane;
            if (i < n_in) lanes16[lane] = pim_func::f32_to_f16(x[i]);
        }
        aim_gpr_write_bytes(static_cast<uint32_t>(col), 0, lanes16, 32);
    }
}

int issue_one(const PimHybridCmd &c) {
    pim_func::HardwareInfo hw = pim_func::gem5_hw();
    const uint64_t isr_mask = aim_func_to_isr_chmask(hw.pim_chmask, hw.n_channels);
    uint64_t mask = isr_mask;
    if (c.opcode == static_cast<uint32_t>(pim_func::Opcode::RD_MAC) && c.chmask > 0 &&
        (c.chmask & (c.chmask - 1)) == 0) {
        mask = aim_func_to_isr_chmask(static_cast<uint64_t>(c.chmask), hw.n_channels);
    }
    const int32_t opsize =
        (c.opcode == static_cast<uint32_t>(pim_func::Opcode::WR_GB) ||
         c.opcode == static_cast<uint32_t>(pim_func::Opcode::MAC_ABK))
            ? c.opsize
            : -1;
    const int32_t row = (c.row >= 0) ? c.row : -1;
    const uint64_t gpr0 = (c.gpr0 >= 0) ? static_cast<uint64_t>(c.gpr0) : 0;
    if (c.opcode == static_cast<uint32_t>(pim_func::Opcode::EOC) ||
        c.opcode == static_cast<uint32_t>(pim_func::Opcode::SYNC))
        return 0;
    aim_iss_cmd(c.opcode, opsize, gpr0, mask, row);
    return 0;
}

} // namespace

extern "C" {

int pim_hybrid_active(void) { return g_hdr && g_hdr->magic == PIM_HYBRID_MAGIC; }

void *pim_hybrid_base(void) { return g_base; }

PimHybridHeader *pim_hybrid_header(void) { return g_hdr; }

uint64_t pim_hybrid_v0(void) { return PIM_HYBRID_V0; }

uint64_t pim_hybrid_size(void) { return g_size ? g_size : env_size(PIM_HYBRID_DEFAULT_SIZE); }

int pim_hybrid_create_server(uint64_t size) {
    if (g_hdr) return 0;
    if (size < sizeof(PimHybridHeader) + 4096) size = env_size(PIM_HYBRID_DEFAULT_SIZE);
    const std::string name = shm_name();
    shm_unlink(name.c_str());
    int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
    if (fd < 0) {
        std::fprintf(stderr, "pim_hybrid: shm_open %s: %s\n", name.c_str(), std::strerror(errno));
        return -1;
    }
    if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        std::fprintf(stderr, "pim_hybrid: ftruncate: %s\n", std::strerror(errno));
        close(fd);
        return -1;
    }
    g_fd = fd;
    g_server = 1;
    if (mmap_at_v0(fd, size, 1) != 0) {
        close(fd);
        g_fd = -1;
        return -1;
    }
    std::fprintf(stderr, "pim_hybrid: server V0=%#llx size=%llu shm=%s\n",
                 (unsigned long long)PIM_HYBRID_V0, (unsigned long long)size, name.c_str());
    return 0;
}

int pim_hybrid_attach_client(void) {
    if (g_hdr && g_hdr->magic == PIM_HYBRID_MAGIC) {
        g_hdr->a_ready = 1;
        return 0;
    }
    const std::string name = shm_name();
    int fd = -1;
    for (int i = 0; i < 6000; i++) {
        fd = shm_open(name.c_str(), O_RDWR, 0600);
        if (fd >= 0) break;
        usleep(10000);
    }
    if (fd < 0) {
        std::fprintf(stderr, "pim_hybrid: client shm_open %s timed out\n", name.c_str());
        return -1;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return -1;
    }
    g_fd = fd;
    if (mmap_at_v0(fd, static_cast<uint64_t>(st.st_size), 0) != 0) {
        close(fd);
        g_fd = -1;
        return -1;
    }
    for (int i = 0; i < 2000 && (!g_hdr->b_ready || g_hdr->magic != PIM_HYBRID_MAGIC); i++)
        usleep(1000);
    if (g_hdr->magic != PIM_HYBRID_MAGIC || !g_hdr->b_ready) {
        std::fprintf(stderr, "pim_hybrid: client magic/b_ready wait failed\n");
        return -1;
    }
    g_hdr->a_ready = 1;
    std::fprintf(stderr, "pim_hybrid: client attached V0=%#llx size=%llu\n",
                 (unsigned long long)g_hdr->v0, (unsigned long long)g_hdr->size);
    return 0;
}

int pim_hybrid_guest_attach(void) {
    g_base = reinterpret_cast<void *>(PIM_HYBRID_V0);
    g_hdr = reinterpret_cast<PimHybridHeader *>(g_base);
    for (int i = 0; i < 100000 && g_hdr->magic != PIM_HYBRID_MAGIC; i++) {
    }
    if (g_hdr->magic != PIM_HYBRID_MAGIC) return -1;
    g_size = g_hdr->size;
    return 0;
}

void *pim_hybrid_alloc(size_t n) {
    if (!g_hdr || n == 0) return nullptr;
    uint64_t need = align_up(static_cast<uint64_t>(n), kAlign);
    uint64_t off = g_hdr->bump;
    if (off + need > g_hdr->size) {
        std::fprintf(stderr, "pim_hybrid: arena full bump=%llu need=%llu size=%llu\n",
                     (unsigned long long)off, (unsigned long long)need,
                     (unsigned long long)g_hdr->size);
        return nullptr;
    }
    g_hdr->bump = off + need;
    void *p = reinterpret_cast<uint8_t *>(g_base) + off;
    std::memset(p, 0, static_cast<size_t>(need));
    return p;
}

int pim_hybrid_contains(const void *p) {
    if (!g_base || !p) return 0;
    auto a = reinterpret_cast<uintptr_t>(p);
    auto b = reinterpret_cast<uintptr_t>(g_base);
    return a >= b && a < b + g_size;
}

int pim_hybrid_record_region(int ch, int n_banks, int row0, int n_rows, int col0, int n_cols,
                             void *data) {
    if (!g_hdr || !data) return -1;
    if (g_hdr->n_regions >= PIM_HYBRID_MAX_REGIONS) return -1;
    const uint32_t i = g_hdr->n_regions;
    PimHybridRegion &r = g_hdr->regions[i];
    r.ch = ch;
    r.n_banks = n_banks;
    r.row0 = row0;
    r.n_rows = n_rows;
    r.col0 = col0;
    r.n_cols = n_cols;
    r.data_off = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data) -
                                       reinterpret_cast<uintptr_t>(g_base));
    g_hdr->n_regions = i + 1;
    return 0;
}

void pim_hybrid_start_mac_worker(void) {
    if (g_mac_started || !g_hdr) return;
    g_mac_started = 1;
    if (pthread_create(&g_mac_th, nullptr, mac_worker_main, nullptr) != 0) {
        g_mac_started = 0;
        std::fprintf(stderr, "pim_hybrid: mac worker spawn failed\n");
    }
}

void pim_hybrid_stop_mac_worker(void) {
    if (!g_hdr || !g_mac_started) return;
    g_hdr->stop = 1;
    futex_wake((uint32_t *)&g_hdr->gemv_seq, 8);
    pthread_join(g_mac_th, nullptr);
    g_mac_started = 0;
}

int pim_hybrid_submit(const float *x, float *y, int n_out, int n_in, int row_base, int n_cols,
                      int acc, const PimHybridCmd *cmds, int n_cmds) {
    if (!g_hdr || !x || !y || n_out <= 0 || n_in <= 0) return -1;
    const int n_in_pad = n_cols * pim_func::kLanes;
    if (n_in_pad > PIM_HYBRID_X_MAX || n_out > PIM_HYBRID_Y_MAX) return -1;
    uint8_t *scratch = reinterpret_cast<uint8_t *>(g_base) +
                       align_up(sizeof(PimHybridHeader), kAlign);
    float *x_a = reinterpret_cast<float *>(scratch);
    float *y_a = x_a + PIM_HYBRID_X_MAX;
    std::memcpy(x_a, x, static_cast<size_t>(n_in) * sizeof(float));
    if (n_in_pad > n_in) {
        std::memset(x_a + n_in, 0, static_cast<size_t>(n_in_pad - n_in) * sizeof(float));
    }
    if (acc) std::memcpy(y_a, y, static_cast<size_t>(n_out) * sizeof(float));
    const int n_banks = pim_func::kBanks;
    const int n_row_groups = (n_out + n_banks - 1) / n_banks;
    pim_func::HardwareInfo hw = pim_func::gem5_hw();
    pim_func::CentGemvAllocation alloc;
    const auto chs = alloc.pim_channels(hw);
    const int n_ch = static_cast<int>(chs.size()) > 0 ? static_cast<int>(chs.size()) : 1;
    const int n_waves = (n_row_groups + n_ch - 1) / n_ch;

    if (n_cmds > 0 && cmds) {
        if (n_cmds > PIM_HYBRID_MAX_CMDS) n_cmds = PIM_HYBRID_MAX_CMDS;
        std::memcpy(g_hdr->cmds, cmds, static_cast<size_t>(n_cmds) * sizeof(PimHybridCmd));
        g_hdr->n_cmds = static_cast<uint32_t>(n_cmds);
    } else {
        g_hdr->n_cmds = 0;
    }

    g_hdr->n_out = n_out;
    g_hdr->n_in = n_in;
    g_hdr->row_base = row_base;
    g_hdr->n_cols = n_cols;
    g_hdr->acc = acc;
    g_hdr->n_waves = n_waves;
    g_hdr->n_banks = n_banks;
    g_hdr->ch = chs.empty() ? 0 : chs[0];
    g_hdr->x_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(x_a));
    g_hdr->y_ptr = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(y_a));

    const uint32_t seq = g_hdr->gemv_seq + 1;
    std::atomic_thread_fence(std::memory_order_release);
    g_hdr->isr_seq = seq;
    g_hdr->gemv_seq = seq;
    futex_wake((uint32_t *)&g_hdr->gemv_seq, 1);

    while (g_hdr->gemv_done != seq) {
        futex_wait((uint32_t *)&g_hdr->gemv_done, g_hdr->gemv_done);
    }
    if (g_hdr->n_cmds > 0) {
        while (g_hdr->isr_done != seq) sched_yield();
    }
    if (y_a != y) std::memcpy(y, y_a, static_cast<size_t>(n_out) * sizeof(float));
    return 0;
}

void pim_hybrid_request_stop(void) {
    if (!g_hdr) return;
    g_hdr->stop = 1;
    futex_wake((uint32_t *)&g_hdr->gemv_seq, 8);
    futex_wake((uint32_t *)&g_hdr->gemv_done, 8);
}

int pim_hybrid_guest_loop(void) {
    if (pim_hybrid_guest_attach() != 0) {
        std::fprintf(stderr, "pim_hybrid: guest attach failed\n");
        return 1;
    }
    PimHybridHeader *h = g_hdr;
    uint32_t seen = 0;
    std::fprintf(stderr, "pim_hybrid: guest loop at V0 magic=%llx a_ready=%u\n",
                 (unsigned long long)h->magic, h->a_ready);
    while (!h->stop) {
        uint32_t seq = h->isr_seq;
        if (seq == seen) {
            /* Avoid uncacheable PIO + unimplemented nanosleep in gem5 SE. */
            for (int i = 0; i < 4096; i++)
                __asm__ volatile("nop");
            continue;
        }
        seen = seq;
        const float *x = reinterpret_cast<const float *>(static_cast<uintptr_t>(h->x_ptr));
        if (x && h->n_in > 0 && h->n_cols > 0) {
            mmio_pack_x_from(x, h->n_in, h->n_cols, pim_func::kLanes);
            uint8_t z[32] = {};
            aim_gpr_write_bytes(static_cast<uint32_t>(h->n_cols), 0, z, 32);
        }
        for (uint32_t i = 0; i < h->n_cmds; i++) issue_one(h->cmds[i]);
        std::atomic_thread_fence(std::memory_order_release);
        h->isr_done = seq;
    }
    return 0;
}

} // extern "C"

namespace pim_func {

void hybrid_sync_dram(BankedDram *dram) {
    if (!dram || !g_hdr) return;
    dram->adopt_hybrid(g_base, g_hdr->regions, g_hdr->n_regions);
}

} // namespace pim_func
