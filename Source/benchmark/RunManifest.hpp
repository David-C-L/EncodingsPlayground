#pragma once

// Provenance sidecar written next to every result file.
//
// The manifest exists because a timing table is uninterpretable on its own.  A
// cold-path number measured under `powersave` is not comparable with one
// measured under `performance`; a TLB-sensitive number taken with transparent
// huge pages `always` is not comparable with a 4 KiB-page one; and neither is
// recoverable after the fact.  So the environment is recorded at run time, next
// to the numbers it produced.
//
// Two rules follow from that purpose.
//
//  * Nothing here may throw or abort a sweep.  Every probe degrades to the
//    string "unknown": provenance that can kill a six-hour run is worse than
//    provenance that is occasionally incomplete.
//  * Build facts come from macros stamped in at configure time (see
//    Source/benchmark/CMakeLists.txt), never from shelling out to git at run
//    time — the working tree may have moved on since the binary was built, and
//    the commit that matters is the one it was compiled from.

#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include <limits.h>
#include <unistd.h>

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

namespace encodings::benchmark {

namespace detail {

/// First line of `path`, trimmed; "unknown" if it cannot be read.  Deliberately
/// swallows everything: an unreadable /sys node is normal in a container.
inline std::string readFirstLine(const std::filesystem::path& path) {
    try {
        std::ifstream in(path);
        std::string   line;
        if (in && std::getline(in, line)) {
            const size_t b = line.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) return "unknown";
            const size_t e = line.find_last_not_of(" \t\r\n");
            return line.substr(b, e - b + 1);
        }
    } catch (...) {
    }
    return "unknown";
}

inline std::string jsonEscape(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char ch : s) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (ch < 0x20) {
                    // \u escape rather than a raw control byte: CPU model strings and
                    // argv have both been seen to carry them, and a raw one makes the
                    // whole manifest unparseable.
                    static constexpr char kHex[] = "0123456789abcdef";
                    out += "\\u00";
                    out += kHex[(ch >> 4) & 0xF];
                    out += kHex[ch & 0xF];
                } else {
                    out += static_cast<char>(ch);
                }
        }
    }
    return out;
}

inline std::string isoNow() {
    const std::time_t t = std::time(nullptr);
    std::tm           tm{};
#if defined(_WIN32)
    if (gmtime_s(&tm, &t) != 0) return "unknown";
#else
    if (gmtime_r(&t, &tm) == nullptr) return "unknown";
#endif
    char buf[32];
    if (std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm) == 0) return "unknown";
    return std::string(buf);
}

}  // namespace detail

struct RunManifest {
    std::string driverName{"unknown"};
    std::string startedAtIso{"unknown"};
    std::string finishedAtIso{"unknown"};
    int         exitCode{0};

    std::string gitSha{"unknown"};
    std::string gitDirty{"unknown"};
    std::string cxxFlags{"unknown"};
    std::string buildType{"unknown"};
    std::string compiler{"unknown"};

    std::string hostname{"unknown"};
    std::string cpuModel{"unknown"};
    std::string openzlEnabled{"unknown"};
    size_t      onlineCores{0};

    // Plain numbers, filled in by the caller from whatever detected the topology.
    // This header deliberately does not depend on the cache-policy code: a
    // manifest must be writable by a driver that never touches a cache probe.
    size_t l1dBytes{0};
    size_t l2Bytes{0};
    size_t llcBytes{0};
    size_t lineBytes{0};

    std::string scalingGovernor{"unknown"};
    std::string smtEnabled{"unknown"};
    std::string thpEnabled{"unknown"};

    uint64_t                 seed{0};
    std::vector<std::string> argv;
    std::vector<std::string> encoders;
    std::vector<std::string> datasets;

    /// path -> "size=N,mtime=T".  Cheap enough to take per run, and enough to
    /// notice that a dataset was regenerated between two sweeps being compared.
    std::map<std::string, std::string> datasetFingerprints;
    std::map<std::string, std::string> extra;

    void setCacheSizes(size_t l1d, size_t l2, size_t llc, size_t line) {
        l1dBytes  = l1d;
        l2Bytes   = l2;
        llcBytes  = llc;
        lineBytes = line;
    }

    /// stat()s the file and records size and mtime; records "unknown" if it is
    /// gone, which is itself worth knowing.
    void fingerprintDataset(const std::filesystem::path& path) {
        std::string value = "unknown";
        try {
            if (std::filesystem::exists(path)) {
                const auto size = std::filesystem::file_size(path);
                // Unix seconds, not the raw file_clock tick count: file_clock's epoch
                // is implementation-defined (negative on libstdc++), so the raw count
                // is not comparable against anything outside this process.
                const auto mtime = std::chrono::duration_cast<std::chrono::seconds>(
                                       std::chrono::clock_cast<std::chrono::system_clock>(
                                           std::filesystem::last_write_time(path))
                                           .time_since_epoch())
                                       .count();
                value = "size=" + std::to_string(size) + ",mtime=" + std::to_string(mtime);
            }
        } catch (...) {
        }
        datasetFingerprints[path.string()] = value;
    }

    static RunManifest capture(std::string driverName, const std::vector<std::string>& argv) {
        RunManifest m;
        m.driverName   = std::move(driverName);
        m.startedAtIso = detail::isoNow();
        m.argv         = argv;

        // Guarded so this header still compiles in a standalone TU without the
        // encodings_benchmark target's compile definitions.
#ifdef ENCODINGS_GIT_SHA
        m.gitSha = ENCODINGS_GIT_SHA;
#endif
#ifdef ENCODINGS_GIT_DIRTY
        m.gitDirty = ENCODINGS_GIT_DIRTY;
#endif
#ifdef ENCODINGS_CXX_FLAGS
        m.cxxFlags = ENCODINGS_CXX_FLAGS;
#endif
#ifdef ENCODINGS_BUILD_TYPE
        m.buildType = ENCODINGS_BUILD_TYPE;
#endif
#if defined(__clang__)
        m.compiler = "clang " __clang_version__;
#elif defined(__GNUC__)
        m.compiler = "gcc " __VERSION__;
#endif
#ifdef HAVE_OPENZL
        m.openzlEnabled = "1";
#else
        m.openzlEnabled = "0";
#endif

        char host[HOST_NAME_MAX + 1] = {};
        m.hostname = (::gethostname(host, sizeof(host) - 1) == 0 && host[0] != '\0')
                         ? std::string(host)
                         : std::string("unknown");

        // One pass over /proc/cpuinfo for both facts.  onlineCores counts
        // "processor" lines, i.e. online logical CPUs — which is what a thread
        // count in a driver has to be checked against, and is not the same as
        // hardware_concurrency() under a cgroup CPU limit.
        try {
            std::ifstream in("/proc/cpuinfo");
            std::string   line;
            while (in && std::getline(in, line)) {
                if (line.starts_with("processor")) {
                    ++m.onlineCores;
                } else if (m.cpuModel == "unknown" && line.starts_with("model name")) {
                    if (const size_t colon = line.find(':'); colon != std::string::npos) {
                        const size_t b = line.find_first_not_of(" \t", colon + 1);
                        if (b != std::string::npos) m.cpuModel = line.substr(b);
                    }
                }
            }
        } catch (...) {
        }

        m.scalingGovernor =
            detail::readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor");
        m.smtEnabled = detail::readFirstLine("/sys/devices/system/cpu/smt/control");
        m.thpEnabled = detail::readFirstLine("/sys/kernel/mm/transparent_hugepage/enabled");
        return m;
    }

    /// Writes `<outputPath>.manifest.json`, truncating any previous content.
    /// Idempotent by construction, because drivers call it twice: once before the
    /// sweep so a killed run still has provenance, once after with the finish time
    /// and exit code.
    void writeSidecar(const std::filesystem::path& outputPath) const {
        const std::filesystem::path sidecar =
            std::filesystem::path(outputPath.string() + ".manifest.json");
        try {
            if (sidecar.has_parent_path())
                std::filesystem::create_directories(sidecar.parent_path());
            std::ofstream out(sidecar, std::ios::out | std::ios::trunc);
            if (!out) return;

            out << "{\n";
            emitString(out, "driver", driverName);
            emitString(out, "started_at", startedAtIso);
            emitString(out, "finished_at", finishedAtIso);
            out << "  \"exit_code\": " << exitCode << ",\n";
            emitString(out, "git_sha", gitSha);
            emitString(out, "git_dirty", gitDirty);
            emitString(out, "build_type", buildType);
            emitString(out, "cxx_flags", cxxFlags);
            emitString(out, "compiler", compiler);
            emitString(out, "openzl_enabled", openzlEnabled);
            emitString(out, "hostname", hostname);
            emitString(out, "cpu_model", cpuModel);
            out << "  \"online_cores\": " << onlineCores << ",\n";
            out << "  \"cache\": {\"l1d_bytes\": " << l1dBytes << ", \"l2_bytes\": " << l2Bytes
                << ", \"llc_bytes\": " << llcBytes << ", \"line_bytes\": " << lineBytes
                << "},\n";
            emitString(out, "scaling_governor", scalingGovernor);
            emitString(out, "smt", smtEnabled);
            emitString(out, "thp", thpEnabled);
            out << "  \"seed\": " << seed << ",\n";
            emitArray(out, "argv", argv);
            emitArray(out, "encoders", encoders);
            emitArray(out, "datasets", datasets);
            emitObject(out, "dataset_fingerprints", datasetFingerprints, /*last=*/false);
            emitObject(out, "extra", extra, /*last=*/true);
            out << "}\n";
        } catch (...) {
            // A manifest is provenance, not a hard dependency.
        }
    }

  private:
    static void emitString(std::ostream& out, std::string_view key, std::string_view value) {
        out << "  \"" << key << "\": \"" << detail::jsonEscape(value) << "\",\n";
    }

    static void emitArray(std::ostream& out, std::string_view key,
                          const std::vector<std::string>& values) {
        out << "  \"" << key << "\": [";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i != 0) out << ", ";
            out << '"' << detail::jsonEscape(values[i]) << '"';
        }
        out << "],\n";
    }

    static void emitObject(std::ostream& out, std::string_view key,
                           const std::map<std::string, std::string>& values, bool last) {
        out << "  \"" << key << "\": {";
        bool first = true;
        for (const auto& [k, v] : values) {
            if (!first) out << ", ";
            first = false;
            out << '"' << detail::jsonEscape(k) << "\": \"" << detail::jsonEscape(v) << '"';
        }
        out << '}' << (last ? "\n" : ",\n");
    }
};

}  // namespace encodings::benchmark
