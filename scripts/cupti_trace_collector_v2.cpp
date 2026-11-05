// Minimal CUPTI 12.9 activity collector; independent of the frozen CUDA11.8 codec.
// Record layouts follow the installed official CUPTI 12.9 activity header.
#include <cupti.h>
#include <windows.h>
#include <malloc.h>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

static std::ofstream output;
static std::mutex output_mutex;
static uint64_t dropped_total = 0;
static std::string last_error;
static std::string first_error;
static CUpti_ActivityKind kinds[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DRIVER,
    CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
    CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION};

static std::string quote(const char* value) {
    std::ostringstream out;
    out << '"';
    if (value) for (const unsigned char* p = (const unsigned char*)value; *p; ++p) {
        if (*p == '"' || *p == '\\') out << '\\' << *p;
        else if (*p == '\n') out << "\\n";
        else if (*p == '\r') out << "\\r";
        else if (*p == '\t') out << "\\t";
        else if (*p < 32) out << '?';
        else out << *p;
    }
    out << '"'; return out.str();
}

static int check(CUptiResult result, const char* label) {
    if (result == CUPTI_SUCCESS) return 0;
    const char* error = nullptr; cuptiGetResultString(result, &error);
    { std::lock_guard<std::mutex> lock(output_mutex);
      last_error = std::string(label) + ": " + (error ? error : "unknown");
      if (first_error.empty()) first_error = last_error;
      if (output.is_open()) output << "{\"cat\":\"collector_error\",\"code\":" << (int)result << ",\"message\":" << quote(last_error.c_str()) << "}\n"; }
    return (int)result;
}

static void CUPTIAPI requested(uint8_t** buffer, size_t* size, size_t* count) {
    *size = 4 * 1024 * 1024; *count = 0;
    *buffer = (uint8_t*)_aligned_malloc(*size, 8);
    if (!*buffer) { *size = 0; check(CUPTI_ERROR_OUT_OF_MEMORY, "activity buffer allocation"); }
}

static void write_record(CUpti_Activity* record) {
    std::ostringstream row;
    row << "{\"kind\":" << (int)record->kind;
    auto timing = [&](uint64_t start, uint64_t end, uint32_t corr) {
        row << ",\"start_ns\":" << start << ",\"end_ns\":" << end << ",\"correlation\":" << corr;
    };
    auto device = [&](uint32_t id, uint32_t context, uint32_t stream) {
        row << ",\"device\":" << id << ",\"context\":" << context << ",\"stream\":" << stream;
    };
    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
        auto p = (CUpti_ActivityKernel9*)record;
        row << ",\"cat\":\"kernel\",\"name\":" << quote(p->name);
        timing(p->start, p->end, p->correlationId); device(p->deviceId, p->contextId, p->streamId);
        row << ",\"grid\":[" << p->gridX << ',' << p->gridY << ',' << p->gridZ
            << "],\"block\":[" << p->blockX << ',' << p->blockY << ',' << p->blockZ << ']';
        break;
    }
    case CUPTI_ACTIVITY_KIND_RUNTIME:
    case CUPTI_ACTIVITY_KIND_DRIVER: {
        auto p = (CUpti_ActivityAPI*)record; const char* name = nullptr;
        check(cuptiGetCallbackName(record->kind == CUPTI_ACTIVITY_KIND_RUNTIME ? CUPTI_CB_DOMAIN_RUNTIME_API : CUPTI_CB_DOMAIN_DRIVER_API, p->cbid, &name), "callback name");
        row << ",\"cat\":" << (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME ? "\"cuda_runtime\"" : "\"cuda_driver\"")
            << ",\"name\":" << quote(name) << ",\"cbid\":" << p->cbid
            << ",\"pid\":" << p->processId << ",\"tid\":" << p->threadId << ",\"return_value\":" << p->returnValue;
        timing(p->start, p->end, p->correlationId); break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
        auto p = (CUpti_ActivityMemcpy6*)record;
        row << ",\"cat\":\"gpu_memcpy\",\"name\":\"CUPTI memcpy\",\"bytes\":" << p->bytes << ",\"copy_kind\":" << (int)p->copyKind;
        timing(p->start, p->end, p->correlationId); device(p->deviceId, p->contextId, p->streamId); break;
    }
    case CUPTI_ACTIVITY_KIND_MEMSET: {
        auto p = (CUpti_ActivityMemset4*)record;
        row << ",\"cat\":\"gpu_memset\",\"name\":\"CUPTI memset\",\"bytes\":" << p->bytes << ",\"value\":" << p->value;
        timing(p->start, p->end, p->correlationId); device(p->deviceId, p->contextId, p->streamId); break;
    }
    case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION: {
        auto p = (CUpti_ActivitySynchronization2*)record;
        row << ",\"cat\":\"cuda_sync\",\"name\":\"CUPTI synchronization\",\"sync_type\":" << (int)p->type
            << ",\"context\":" << p->contextId << ",\"stream\":" << p->streamId;
        timing(p->start, p->end, p->correlationId); break;
    }
    case CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION: {
        auto p = (CUpti_ActivityExternalCorrelation*)record;
        row << ",\"cat\":\"external_correlation\",\"external_id\":" << p->externalId
            << ",\"correlation\":" << p->correlationId << ",\"external_kind\":" << (int)p->externalKind; break;
    }
    default: row << ",\"cat\":\"unhandled\"";
    }
    row << "}\n";
    std::lock_guard<std::mutex> lock(output_mutex); output << row.str();
}

static void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t* buffer, size_t size, size_t valid) {
    CUpti_Activity* record = nullptr;
    CUptiResult result;
    while ((result = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) write_record(record);
    if (result != CUPTI_ERROR_MAX_LIMIT_REACHED) check(result, "cuptiActivityGetNextRecord");
    size_t dropped = 0; check(cuptiActivityGetNumDroppedRecords(context, stream, &dropped), "dropped records");
    {
        std::lock_guard<std::mutex> lock(output_mutex);
        dropped_total += dropped;
        output << "{\"cat\":\"buffer_metadata\",\"valid_bytes\":" << valid << ",\"dropped_records\":" << dropped << "}\n";
    }
    _aligned_free(buffer);
}

#define API extern "C" __declspec(dllexport)
API const char* trace_error() { static thread_local std::string copy; std::lock_guard<std::mutex> lock(output_mutex); copy = last_error; return copy.c_str(); }
API uint64_t trace_timestamp() { uint64_t value = 0; check(cuptiGetTimestamp(&value), "timestamp"); return value; }
API int trace_start(const char* path) {
    output.open(path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!output.is_open()) { last_error = "Cannot create activity output"; return -1; }
    uint32_t version = 0;
    if (check(cuptiGetVersion(&version), "version")) return -1;
    output << "{\"cat\":\"collector_metadata\",\"cupti_version\":" << version << "}\n";
    if (check(cuptiActivityRegisterCallbacks(requested, completed), "register callbacks")) return -1;
    for (auto kind : kinds) {
        if (check(cuptiActivityEnable(kind), "enable activity")) return -1;
        output << "{\"cat\":\"enabled_activity\",\"kind\":" << (int)kind << "}\n";
    }
    return 0;
}
API int trace_push(uint64_t id) { return check(cuptiActivityPushExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, id), "push external id"); }
API int trace_pop() { uint64_t value = 0; return check(cuptiActivityPopExternalCorrelationId(CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0, &value), "pop external id"); }
API int trace_stop() {
    int result = check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "flush all");
    for (auto kind : kinds) if (check(cuptiActivityDisable(kind), "disable activity")) result = -1;
    if (check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED), "final flush")) result = -1;
    std::lock_guard<std::mutex> lock(output_mutex);
    output << "{\"cat\":\"collector_summary\",\"dropped_records_total\":" << dropped_total << ",\"error\":" << quote(last_error.c_str()) << ",\"first_error\":" << quote(first_error.c_str()) << "}\n";
    output.close(); return result;
}
