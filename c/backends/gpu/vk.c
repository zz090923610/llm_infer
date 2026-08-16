#include "vk.h"
#include "util.h"
#include "quant.h"
#include "shaders/shaders_spv.h"
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <string.h>

#define VKCHK(expr)                                                                                 \
    do {                                                                                            \
        VkResult _r = (expr);                                                                       \
        if (_r != VK_SUCCESS) die("vulkan %s failed: %d", #expr, (int)_r);                          \
    } while (0)

#define GPU_MAX_BIND 8
#define GPU_MAX_SETS 1024
#define GPU_INTERN_CAP 1024

struct GpuBuf {
    const void *host;
    size_t bytes;
    GpuBufKind kind;
    VkBuffer buffer;
    VkDeviceMemory memory;
    void *mapped;
    int uploaded;
    int need_upload;
    int gpu_dirty;
    uint64_t fp;
    size_t live; /* bytes the current intern() call is using (prefix of `bytes`) */
    VkBufferView texel_view;
    VkBufferView texel_view_rgba;
};

static int g_inited;
static char g_devname[256];
static char g_backend_name[288];
static VkInstance g_inst;
static VkPhysicalDevice g_phys;
static VkDevice g_dev;
static VkQueue g_queue;
static uint32_t g_qfam;
static VkCommandPool g_pool;
static VkCommandBuffer g_cmd;
static VkFence g_fence;
static VkDescriptorSetLayout g_dsl;
static VkPipelineLayout g_pl;
static VkDescriptorSetLayout g_dsl_tex;
static VkPipelineLayout g_pl_tex;
static VkDescriptorPool g_dpool;
static VkDescriptorSet g_sets[GPU_MAX_SETS];
static uint32_t g_nsets;
static int g_push_desc;
static PFN_vkCmdPushDescriptorSetKHR g_vkCmdPushDescriptorSetKHR;
static VkPipeline g_pipe[PIPE_COUNT];
static VkBuffer g_dummy_buf;
static VkDeviceMemory g_dummy_mem;
static VkBufferView g_dummy_texel;
static int g_recording;
static int g_need_barrier;
static uint32_t g_mem_host;
static VkPhysicalDeviceProperties g_props;
static VkDeviceSize g_ssbo_align;
static int g_f16_texel;
static int g_rgba16_texel;

static GpuBuf g_bufs[GPU_INTERN_CAP];
static int g_nbufs;
static size_t g_weight_bytes;
static size_t g_weight_logged;

static void flush_mapped(GpuBuf *b);
static void invalidate_mapped(GpuBuf *b);

static uint64_t buf_fp(const void *p, size_t n) {
    const unsigned char *b = p;
    uint64_t h = 14695981039346656037ull ^ n;
    if (!n || !p) return h;
    size_t lim = n < 128 ? n : 128;
    for (size_t i = 0; i < lim; i++) {
        h ^= b[i];
        h *= 1099511628211ull;
    }
    if (n > 128) {
        h ^= b[n - 1];
        h *= 1099511628211ull;
    }
    return h;
}

static uint32_t find_mem(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &mp);
    int best = -1, best_score = -1;
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        if (!(bits & (1u << i))) continue;
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & want) != want) continue;
        int score = 0;
        if (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) score += 4;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) score += 1;
        if (score > best_score) {
            best_score = score;
            best = (int)i;
        }
    }
    if (best >= 0) return (uint32_t)best;
    die("no host-visible Vulkan memory type");
    return 0;
}

static void alloc_buf(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer *buf, VkDeviceMemory *mem,
                      void **mapped) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = size < 256 ? 256 : size,
                             .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VKCHK(vkCreateBuffer(g_dev, &bi, NULL, buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(g_dev, *buf, &req);
    VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                               .allocationSize = req.size,
                               .memoryTypeIndex = find_mem(req.memoryTypeBits, g_mem_host)};
    VKCHK(vkAllocateMemory(g_dev, &ai, NULL, mem));
    VKCHK(vkBindBufferMemory(g_dev, *buf, *mem, 0));
    if (mapped) VKCHK(vkMapMemory(g_dev, *mem, 0, req.size, 0, mapped));
}

static VkShaderModule make_shader(const SpvBlob *blob) {
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                   .codeSize = blob->n_words * 4,
                                   .pCode = blob->words};
    VkShaderModule m;
    VKCHK(vkCreateShaderModule(g_dev, &ci, NULL, &m));
    return m;
}

static int pipe_texel_w(int pipe) {
    return pipe == GPU_PIPE_LINEAR || pipe == GPU_PIPE_LINEAR_GEMM || pipe == GPU_PIPE_LINEAR_F16;
}

static VkBufferUsageFlags buf_usage(GpuBufKind kind) {
    VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if (kind == GPU_BUF_WEIGHT || kind == GPU_BUF_WEIGHT_F16)
        u |= VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
    return u;
}

static VkBufferView make_texel_view(VkBuffer buf, VkFormat fmt, VkDeviceSize range) {
    VkBufferViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO,
                                 .buffer = buf,
                                 .format = fmt,
                                 .offset = 0,
                                 .range = range};
    VkBufferView v;
    VKCHK(vkCreateBufferView(g_dev, &ci, NULL, &v));
    return v;
}

static VkBufferView texel_view_w(GpuBuf *b) {
    if (!b || !b->buffer) return g_dummy_texel;
    if (b->texel_view) return b->texel_view;
    VkFormat fmt = VK_FORMAT_R32_SFLOAT;
    uint32_t ntex = (uint32_t)((b->bytes + 3u) / 4u);
    if (b->kind == GPU_BUF_WEIGHT_F16) {
        fmt = VK_FORMAT_R16_SFLOAT;
        ntex = (uint32_t)(b->bytes / sizeof(float));
    }
    if (ntex > g_props.limits.maxTexelBufferElements)
        die("texel buffer %u elements > device max %u", ntex,
            g_props.limits.maxTexelBufferElements);
    b->texel_view = make_texel_view(b->buffer, fmt, VK_WHOLE_SIZE);
    return b->texel_view;
}

static VkBufferView texel_view_rgba16(GpuBuf *b) {
    if (!b || !b->buffer) return g_dummy_texel;
    if (b->texel_view_rgba) return b->texel_view_rgba;
    uint32_t ntex = (uint32_t)(b->bytes / sizeof(float) / 4u);
    if (ntex > g_props.limits.maxTexelBufferElements)
        die("rgba16 texel buffer %u elements > device max %u", ntex,
            g_props.limits.maxTexelBufferElements);
    b->texel_view_rgba = make_texel_view(b->buffer, VK_FORMAT_R16G16B16A16_SFLOAT, VK_WHOLE_SIZE);
    return b->texel_view_rgba;
}

static void create_pipelines(void) {
    for (int i = 0; i < PIPE_COUNT; i++) {
        VkShaderModule sm = make_shader(&spv_blobs[i]);
        VkPipelineLayout pl = pipe_texel_w(i) ? g_pl_tex : g_pl;
        VkComputePipelineCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                      .module = sm,
                      .pName = "main"},
            .layout = pl};
        VkResult _pr = vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &ci, NULL, &g_pipe[i]);
        if (_pr != VK_SUCCESS)
            die("vulkan compute pipeline %d failed: %d", i, (int)_pr);
        vkDestroyShaderModule(g_dev, sm, NULL);
    }
}

static void begin_cmd(void) {
    if (g_recording) return;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VKCHK(vkBeginCommandBuffer(g_cmd, &bi));
    g_recording = 1;
    g_need_barrier = 0;
}

static void submit_wait(int download_rw) {
    if (g_recording) {
        VKCHK(vkEndCommandBuffer(g_cmd));
        g_recording = 0;
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                           .commandBufferCount = 1,
                           .pCommandBuffers = &g_cmd};
        VKCHK(vkResetFences(g_dev, 1, &g_fence));
        VKCHK(vkQueueSubmit(g_queue, 1, &si, g_fence));
        VKCHK(vkWaitForFences(g_dev, 1, &g_fence, VK_TRUE, UINT64_MAX));
        VKCHK(vkResetCommandBuffer(g_cmd, 0));
        VKCHK(vkResetDescriptorPool(g_dev, g_dpool, 0));
        g_nsets = 0;
        g_need_barrier = 0;
    }
    /* download_rw=0: GPU still owns dirty buffers; do not mark them for re-upload
     * from stale host copies. */
    if (!download_rw) return;
    for (int i = 0; i < g_nbufs; i++) {
        GpuBuf *b = &g_bufs[i];
        if (b->kind == GPU_BUF_WEIGHT || b->kind == GPU_BUF_WEIGHT_F16 ||
            b->kind == GPU_BUF_WEIGHT_Q8) continue;
        if (b->kind == GPU_BUF_DEVICE) {
            b->gpu_dirty = 0;
            continue;
        }
        if (b->gpu_dirty && b->mapped && b->host) {
            invalidate_mapped(b);
            memcpy((void *)b->host, b->mapped, b->live ? b->live : b->bytes);
        }
        b->gpu_dirty = 0;
        b->need_upload = 1;
    }
}

void gpu_init(void) {
    if (g_inited) return;

    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                             .pApplicationName = "llm_infer",
                             .apiVersion = VK_API_VERSION_1_0};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                .pApplicationInfo = &app};
    VKCHK(vkCreateInstance(&ici, NULL, &g_inst));

    uint32_t ndev = 0;
    VKCHK(vkEnumeratePhysicalDevices(g_inst, &ndev, NULL));
    if (!ndev)
        die("no Vulkan physical devices (try LD_LIBRARY_PATH=/vendor/lib64:/vendor/lib64/hw)");
    VkPhysicalDevice *devs = xmalloc((size_t)ndev * sizeof(*devs));
    VKCHK(vkEnumeratePhysicalDevices(g_inst, &ndev, devs));
    g_phys = devs[0];
    for (uint32_t i = 0; i < ndev; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[i], &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ||
            p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            g_phys = devs[i];
            break;
        }
    }
    free(devs);
    vkGetPhysicalDeviceProperties(g_phys, &g_props);
    g_ssbo_align = g_props.limits.minStorageBufferOffsetAlignment;
    if (g_ssbo_align < 16) g_ssbo_align = 16;
    snprintf(g_devname, sizeof(g_devname), "%s", g_props.deviceName);
    snprintf(g_backend_name, sizeof(g_backend_name), "gpu (%s)", g_devname);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &nq, NULL);
    VkQueueFamilyProperties *qps = xmalloc((size_t)nq * sizeof(*qps));
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &nq, qps);
    g_qfam = UINT32_MAX;
    for (uint32_t i = 0; i < nq; i++) {
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            g_qfam = i;
            break;
        }
    }
    free(qps);
    if (g_qfam == UINT32_MAX) die("no Vulkan compute queue");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = g_qfam,
                                   .queueCount = 1,
                                   .pQueuePriorities = &prio};
    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(g_phys, NULL, &next, NULL);
    VkExtensionProperties *exts = next ? xmalloc((size_t)next * sizeof(*exts)) : NULL;
    if (next) VKCHK(vkEnumerateDeviceExtensionProperties(g_phys, NULL, &next, exts));
    g_push_desc = 0;
    for (uint32_t i = 0; i < next; i++) {
        if (strcmp(exts[i].extensionName, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME) == 0) {
            g_push_desc = 1;
            break;
        }
    }
    free(exts);
    const char *dev_exts[1];
    uint32_t n_ext = 0;
    if (g_push_desc) dev_exts[n_ext++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci,
                              .enabledExtensionCount = n_ext,
                              .ppEnabledExtensionNames = n_ext ? dev_exts : NULL};
    VKCHK(vkCreateDevice(g_phys, &dci, NULL, &g_dev));
    vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);
    if (g_push_desc) {
        g_vkCmdPushDescriptorSetKHR = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(
            g_dev, "vkCmdPushDescriptorSetKHR");
        if (!g_vkCmdPushDescriptorSetKHR) g_push_desc = 0;
    }

    g_mem_host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                   .queueFamilyIndex = g_qfam};
    VKCHK(vkCreateCommandPool(g_dev, &pci, NULL, &g_pool));
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                       .commandPool = g_pool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                       .commandBufferCount = 1};
    VKCHK(vkAllocateCommandBuffers(g_dev, &cai, &g_cmd));
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                             .flags = VK_FENCE_CREATE_SIGNALED_BIT};
    VKCHK(vkCreateFence(g_dev, &fci, NULL, &g_fence));

    VkDescriptorSetLayoutBinding binds[GPU_MAX_BIND];
    memset(binds, 0, sizeof(binds));
    for (int i = 0; i < GPU_MAX_BIND; i++) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dlci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .flags = g_push_desc
                                                         ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR
                                                         : 0,
                                            .bindingCount = GPU_MAX_BIND,
                                            .pBindings = binds};
    VKCHK(vkCreateDescriptorSetLayout(g_dev, &dlci, NULL, &g_dsl));
    VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                               .offset = 0,
                               .size = sizeof(GpuPC)};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                       .setLayoutCount = 1,
                                       .pSetLayouts = &g_dsl,
                                       .pushConstantRangeCount = 1,
                                       .pPushConstantRanges = &pcr};
    VKCHK(vkCreatePipelineLayout(g_dev, &plci, NULL, &g_pl));

    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    VKCHK(vkCreateDescriptorSetLayout(g_dev, &dlci, NULL, &g_dsl_tex));
    plci.pSetLayouts = &g_dsl_tex;
    VKCHK(vkCreatePipelineLayout(g_dev, &plci, NULL, &g_pl_tex));
    binds[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;

    VkDescriptorPoolSize psz[2] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = GPU_MAX_SETS * GPU_MAX_BIND},
        {.type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, .descriptorCount = GPU_MAX_SETS},
    };
    VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                       .maxSets = GPU_MAX_SETS,
                                       .poolSizeCount = 2,
                                       .pPoolSizes = psz};
    VKCHK(vkCreateDescriptorPool(g_dev, &dpci, NULL, &g_dpool));

    alloc_buf(256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
              &g_dummy_buf, &g_dummy_mem, NULL);
    g_dummy_texel = make_texel_view(g_dummy_buf, VK_FORMAT_R32_SFLOAT, VK_WHOLE_SIZE);
    _Static_assert(PIPE_COUNT == 18, "shaders_spv.h PIPE_COUNT mismatch");
    create_pipelines();
    g_inited = 1;
    VkFormatProperties fmtp;
    vkGetPhysicalDeviceFormatProperties(g_phys, VK_FORMAT_R16_SFLOAT, &fmtp);
    g_f16_texel = !!(fmtp.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT);
    vkGetPhysicalDeviceFormatProperties(g_phys, VK_FORMAT_R16G16B16A16_SFLOAT, &fmtp);
    g_rgba16_texel = !!(fmtp.bufferFeatures & VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT);
    fprintf(stderr, "gpu: %s (api %u.%u%s, texel max %u%s%s)\n", g_devname,
            VK_VERSION_MAJOR(g_props.apiVersion), VK_VERSION_MINOR(g_props.apiVersion),
            g_push_desc ? ", push desc" : "", g_props.limits.maxTexelBufferElements,
            g_f16_texel ? ", r16 texel" : "", g_rgba16_texel ? ", rgba16 texel" : "");
}

const char *gpu_device_name(void) {
    gpu_init();
    return g_backend_name;
}

GpuBuf *gpu_find(const void *host) {
    if (!host) return NULL;
    for (int i = 0; i < g_nbufs; i++)
        if (g_bufs[i].host == host) return &g_bufs[i];
    return NULL;
}

GpuBuf *gpu_find_containing(const void *p, size_t bytes, size_t *off_out) {
    if (!p || bytes == 0) return NULL;
    const char *cp = (const char *)p;
    for (int i = 0; i < g_nbufs; i++) {
        GpuBuf *b = &g_bufs[i];
        if (!b->host) continue;
        if (b->kind == GPU_BUF_WEIGHT || b->kind == GPU_BUF_WEIGHT_F16 ||
            b->kind == GPU_BUF_WEIGHT_Q8)
            continue;
        const char *h = (const char *)b->host;
        if (cp >= h && cp + bytes <= h + b->bytes) {
            size_t off = (size_t)(cp - h);
            if (off % g_ssbo_align != 0) continue;
            if (off_out) *off_out = off;
            return b;
        }
    }
    return NULL;
}

int gpu_buf_is_f16(const GpuBuf *b) {
    return b && b->kind == GPU_BUF_WEIGHT_F16;
}

int gpu_has_f16_texel(void) {
    gpu_init();
    return g_f16_texel;
}

int gpu_has_rgba16_texel(void) {
    gpu_init();
    return g_rgba16_texel;
}

int gpu_buf_is_q8(const GpuBuf *b) {
    return b && b->kind == GPU_BUF_WEIGHT_Q8;
}

static void flush_mapped(GpuBuf *b) {
    if (!b || !b->memory) return;
    VkMappedMemoryRange r = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                             .memory = b->memory,
                             .offset = 0,
                             .size = VK_WHOLE_SIZE};
    vkFlushMappedMemoryRanges(g_dev, 1, &r);
}

static void invalidate_mapped(GpuBuf *b) {
    if (!b || !b->memory) return;
    VkMappedMemoryRange r = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                             .memory = b->memory,
                             .offset = 0,
                             .size = VK_WHOLE_SIZE};
    vkInvalidateMappedMemoryRanges(g_dev, 1, &r);
}

static size_t gpu_alloc_bytes(GpuBufKind kind, size_t host_bytes) {
    if (kind == GPU_BUF_WEIGHT_F16) {
        size_t n = host_bytes / sizeof(float);
        size_t packed = ((n + 3u) & ~3u) * sizeof(uint16_t);
        return packed < 256 ? 256 : packed;
    }
    if (kind == GPU_BUF_WEIGHT_Q8) {
        size_t n = host_bytes / sizeof(float);
        size_t packed = (n / 32u) * 36u;
        return packed < 256 ? 256 : packed;
    }
    return host_bytes;
}

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if ((x & 0x7fffffffu) > 0x7f800000u) return (uint16_t)(sign | 0x7e00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man = (man | 0x800000u) >> (1 - exp);
        return (uint16_t)(sign | ((man + 0x1000u) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((man + 0x1000u) >> 13));
}

static void destroy_buf(GpuBuf *b) {
    if (b->texel_view) {
        vkDestroyBufferView(g_dev, b->texel_view, NULL);
        b->texel_view = VK_NULL_HANDLE;
    }
    if (b->texel_view_rgba) {
        vkDestroyBufferView(g_dev, b->texel_view_rgba, NULL);
        b->texel_view_rgba = VK_NULL_HANDLE;
    }
    if (b->mapped) {
        vkUnmapMemory(g_dev, b->memory);
        b->mapped = NULL;
    }
    if (b->buffer) vkDestroyBuffer(g_dev, b->buffer, NULL);
    if (b->memory) vkFreeMemory(g_dev, b->memory, NULL);
    b->buffer = VK_NULL_HANDLE;
    b->memory = VK_NULL_HANDLE;
}

GpuBuf *gpu_intern(const void *host, size_t bytes, GpuBufKind kind) {
    gpu_init();
    if (!host || bytes == 0) die("gpu_intern: empty buffer");
    GpuBuf *b = gpu_find(host);
    if (b) {
        int size_mismatch = (b->kind == GPU_BUF_WEIGHT || b->kind == GPU_BUF_WEIGHT_F16 ||
                             b->kind == GPU_BUF_WEIGHT_Q8)
                                ? (bytes != b->bytes)
                                : (bytes > b->bytes);
        if (kind != b->kind || size_mismatch) {
            submit_wait(1);
            destroy_buf(b);
            memset(b, 0, sizeof(*b));
            b->host = host;
            b->bytes = bytes;
            b->live = bytes;
            b->kind = kind;
            b->need_upload = 1;
            alloc_buf(gpu_alloc_bytes(kind, bytes), buf_usage(kind), &b->buffer, &b->memory,
                      &b->mapped);
        }
        b->live = bytes;
        return b;
    }
    if (g_nbufs >= GPU_INTERN_CAP) die("gpu intern table full");
    b = &g_bufs[g_nbufs++];
    memset(b, 0, sizeof(*b));
    b->host = host;
    b->bytes = bytes;
    b->live = bytes;
    b->kind = kind;
    b->need_upload = 1;
    alloc_buf(gpu_alloc_bytes(kind, bytes), buf_usage(kind), &b->buffer, &b->memory, &b->mapped);
    if (kind == GPU_BUF_WEIGHT || kind == GPU_BUF_WEIGHT_F16 || kind == GPU_BUF_WEIGHT_Q8) {
        g_weight_bytes += bytes;
        if (g_weight_bytes >= g_weight_logged + (256ull << 20) || g_weight_logged == 0) {
            g_weight_logged = g_weight_bytes;
            fprintf(stderr, "gpu: interned weights %.1f MiB\n", (double)g_weight_bytes / (1024.0 * 1024.0));
        }
    }
    return b;
}

void gpu_upload(GpuBuf *b) {
    if (!b) return;
    uint64_t fp = buf_fp(b->host, b->bytes);
    if ((b->kind == GPU_BUF_WEIGHT || b->kind == GPU_BUF_WEIGHT_F16 || b->kind == GPU_BUF_WEIGHT_Q8) &&
        b->uploaded && b->fp == fp)
        return;
    if (b->kind == GPU_BUF_DEVICE && b->uploaded) return;
    if (b->kind == GPU_BUF_RW && !b->need_upload && b->fp == fp) return;
    if (b->kind == GPU_BUF_WEIGHT_Q8) return;
    if (b->kind == GPU_BUF_WEIGHT_F16) {
        const float *src = (const float *)b->host;
        uint16_t *dst = (uint16_t *)b->mapped;
        size_t n = b->bytes / sizeof(float);
        for (size_t i = 0; i < n; i++) dst[i] = f32_to_f16(src[i]);
        size_t pad = (n + 3u) & ~3u;
        for (size_t i = n; i < pad; i++) dst[i] = 0;
    } else {
        memcpy(b->mapped, b->host, b->live ? b->live : b->bytes);
    }
    flush_mapped(b);
    b->uploaded = 1;
    b->need_upload = 0;
    b->fp = fp;
}

void gpu_intern_q8(const void *host, const void *q8_blob, int n_elements) {
    if (!host || !q8_blob || n_elements <= 0) return;
    if (n_elements % 32 != 0) {
        GpuBuf *b = gpu_intern(host, (size_t)n_elements * sizeof(float), GPU_BUF_WEIGHT);
        gpu_upload(b);
        return;
    }
    GpuBuf *b = gpu_intern(host, (size_t)n_elements * sizeof(float), GPU_BUF_WEIGHT_Q8);
    const unsigned char *src = (const unsigned char *)q8_blob;
    unsigned char *dst = (unsigned char *)b->mapped;
    int nb = n_elements / 32;
    for (int i = 0; i < nb; i++) {
        const unsigned char *blk = src + (size_t)i * BLOCK_Q8_0;
        uint16_t h;
        memcpy(&h, blk, 2);
        float d = fp16_to_fp32(h);
        memcpy(dst, &d, 4);
        memcpy(dst + 4, blk + 2, 32);
        dst += 36;
    }
    flush_mapped(b);
    b->uploaded = 1;
    b->need_upload = 0;
    b->fp = buf_fp(host, b->bytes);
}

void gpu_wrote(GpuBuf *b) {
    if (!b) return;
    b->gpu_dirty = 1;
    b->need_upload = 0;
    b->uploaded = 1;
}

void gpu_commit(GpuBuf *b) {
    submit_wait(0);
    if (!b) return;
    if (b->mapped && b->host) {
        invalidate_mapped(b);
        memcpy((void *)b->host, b->mapped, b->live ? b->live : b->bytes);
    }
    b->gpu_dirty = 0;
    if (b->kind == GPU_BUF_RW) b->need_upload = 1;
}

void gpu_host_read(const void *p) {
    GpuBuf *b = gpu_find(p);
    if (b && b->kind != GPU_BUF_WEIGHT && b->kind != GPU_BUF_WEIGHT_F16 &&
        b->kind != GPU_BUF_WEIGHT_Q8) gpu_commit(b);
}

void gpu_host_write(void *p) {
    GpuBuf *b = gpu_find(p);
    if (!b) return;
    b->need_upload = 1;
    b->gpu_dirty = 0;
}

void gpu_sync(void) {
    gpu_init();
    submit_wait(1);
}

static void bind_set(GpuBuf **bufs, const size_t *offs, int nbuf, int texel_w, int texel_rgba,
                     VkPipelineLayout pl) {
    VkDescriptorBufferInfo infos[GPU_MAX_BIND];
    VkWriteDescriptorSet writes[GPU_MAX_BIND];
    VkBufferView tview = g_dummy_texel;
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < GPU_MAX_BIND; i++) {
        GpuBuf *b = (i < nbuf) ? bufs[i] : NULL;
        infos[i].buffer = b ? b->buffer : g_dummy_buf;
        infos[i].offset = (b && offs) ? (VkDeviceSize)offs[i] : 0;
        infos[i].range = VK_WHOLE_SIZE;
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding = (uint32_t)i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &infos[i];
    }
    if (texel_w) {
        tview = texel_rgba ? texel_view_rgba16(nbuf > 0 ? bufs[0] : NULL)
                           : texel_view_w(nbuf > 0 ? bufs[0] : NULL);
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        writes[0].pBufferInfo = NULL;
        writes[0].pTexelBufferView = &tview;
    }
    if (g_push_desc) {
        g_vkCmdPushDescriptorSetKHR(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, GPU_MAX_BIND,
                                    writes);
        return;
    }
    VkDescriptorSetLayout dsl = texel_w ? g_dsl_tex : g_dsl;
    VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                      .descriptorPool = g_dpool,
                                      .descriptorSetCount = 1,
                                      .pSetLayouts = &dsl};
    VKCHK(vkAllocateDescriptorSets(g_dev, &ai, &g_sets[g_nsets]));
    VkDescriptorSet set = g_sets[g_nsets++];
    for (int i = 0; i < GPU_MAX_BIND; i++) writes[i].dstSet = set;
    vkUpdateDescriptorSets(g_dev, GPU_MAX_BIND, writes, 0, NULL);
    vkCmdBindDescriptorSets(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &set, 0, NULL);
}

void gpu_dispatch_offs(int pipe, GpuBuf **bufs, const size_t *offs, int nbuf, const GpuPC *pc,
                       uint32_t gx, uint32_t gy, uint32_t gz) {
    gpu_init();
    if (gx == 0 || gy == 0 || gz == 0) return;
    for (int i = 0; i < nbuf; i++) gpu_upload(bufs[i]);
    if (!g_push_desc && g_nsets >= GPU_MAX_SETS) submit_wait(0);
    begin_cmd();
    if (g_need_barrier) {
        VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                              .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                              .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
        vkCmdPipelineBarrier(g_cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        g_need_barrier = 0;
    }
    int texel_w = pipe_texel_w(pipe);
    int texel_rgba = pipe == GPU_PIPE_LINEAR_F16;
    VkPipelineLayout pl = texel_w ? g_pl_tex : g_pl;
    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe[pipe]);
    bind_set(bufs, offs, nbuf, texel_w, texel_rgba, pl);
    vkCmdPushConstants(g_cmd, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(*pc), pc);
    vkCmdDispatch(g_cmd, gx, gy, gz);
    g_need_barrier = 1;
}

void gpu_dispatch(int pipe, GpuBuf **bufs, int nbuf, const GpuPC *pc, uint32_t gx, uint32_t gy,
                  uint32_t gz) {
    gpu_dispatch_offs(pipe, bufs, NULL, nbuf, pc, gx, gy, gz);
}

void gpu_dispatch_1d(int pipe, GpuBuf **bufs, int nbuf, const GpuPC *pc, uint32_t n_threads) {
    if (n_threads == 0) return;
    uint32_t groups = (n_threads + 63u) / 64u;
    uint32_t gx = groups > 65535u ? 65535u : groups;
    uint32_t gy = (groups + gx - 1u) / gx;
    gpu_dispatch(pipe, bufs, nbuf, pc, gx, gy, 1);
}
