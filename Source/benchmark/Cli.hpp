#pragma once

// Table-driven command-line parsing for the benchmark drivers.
//
// The one structural rule here is that an option BINDS TO THE CALLER'S VARIABLE
// and its default is read out of that variable at registration time.  The
// drivers this replaces carried a hand-written help string, which had already
// drifted from the code it documented (defaults changed, flags renamed), and a
// benchmark whose --help lies about its defaults produces results nobody can
// reconstruct.  Here there is exactly one place a default can live, so usage()
// cannot disagree with the parse.

#include <cerrno>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace encodings::benchmark {

class ArgParser {
  public:
    explicit ArgParser(std::string programName, std::string description)
        : programName_(std::move(programName)), description_(std::move(description)) {}

    /// Section header in --help.  Purely presentational; grouping is not scoping,
    /// flags live in one flat namespace whatever group they were declared under.
    ArgParser& group(std::string_view title) {
        Entry e;
        e.kind = Entry::Kind::Group;
        e.text = std::string(title);
        entries_.push_back(std::move(e));
        return *this;
    }

    ArgParser& opt(std::string_view flag, size_t& dst, std::string_view help) {
        return addValued(flag, help, std::format("{}", dst), "N",
                         [&dst](std::string_view v, std::string& err) {
                             return parseUnsigned(v, err, dst);
                         });
    }

    ArgParser& opt(std::string_view flag, double& dst, std::string_view help) {
        return addValued(flag, help, std::format("{}", dst), "X",
                         [&dst](std::string_view v, std::string& err) {
                             return parseDouble(v, err, dst);
                         });
    }

    /// uint64_t overload, present only where it is a DISTINCT type from size_t.
    /// On LP64 the two are the same type and a second declaration would not
    /// compile; the constraint makes it drop out there, where the size_t overload
    /// already binds a `uint64_t&` exactly.
    template <typename U>
        requires std::same_as<U, uint64_t> && (!std::same_as<uint64_t, size_t>)
    ArgParser& opt(std::string_view flag, U& dst, std::string_view help) {
        return addValued(flag, help, std::format("{}", dst), "N",
                         [&dst](std::string_view v, std::string& err) {
                             size_t tmp = 0;
                             if (!parseUnsigned(v, err, tmp)) return false;
                             dst = static_cast<uint64_t>(tmp);
                             return true;
                         });
    }

    ArgParser& opt(std::string_view flag, std::string& dst, std::string_view help) {
        return addValued(flag, help, dst.empty() ? std::string("\"\"") : dst, "STR",
                         [&dst](std::string_view v, std::string&) {
                             dst = std::string(v);
                             return true;
                         });
    }

    ArgParser& opt(std::string_view flag, std::filesystem::path& dst, std::string_view help) {
        return addValued(flag, help, dst.empty() ? std::string("\"\"") : dst.string(), "PATH",
                         [&dst](std::string_view v, std::string&) {
                             dst = std::filesystem::path(v);
                             return true;
                         });
    }

    /// Presence-only switch.  Deliberately has no `--no-x` counterpart: every
    /// driver flag defaults to false, so a negation would only ever restate the
    /// default and give two spellings of the same run.
    ArgParser& flag(std::string_view flagName, bool& dst, std::string_view help) {
        Entry e;
        e.kind        = Entry::Kind::Flag;
        e.flag        = normalize(flagName);
        e.help        = std::string(help);
        e.defaultText = dst ? "true" : "false";
        e.set         = [&dst](std::string_view, std::string&) {
            dst = true;
            return true;
        };
        entries_.push_back(std::move(e));
        return *this;
    }

    /// Accumulating option: each occurrence appends.  The bound vector is NOT
    /// cleared on first use, so a caller-seeded default survives — seed it empty
    /// if "given nothing means everything" is the intent (it is, for filters).
    ArgParser& repeated(std::string_view flag, std::vector<std::string>& dst,
                        std::string_view help) {
        return addValued(flag, help, joined(dst), "STR",
                         [&dst](std::string_view v, std::string&) {
                             dst.emplace_back(v);
                             return true;
                         });
    }

    /// Comma-separated integer list, e.g. `--targets 10,100,1000`.
    /// One occurrence replaces the bound vector rather than appending, so the
    /// caller-supplied default cannot leak into an explicitly given axis.
    ArgParser& list(std::string_view flag, std::vector<size_t>& dst, std::string_view help) {
        return addValued(flag, help, joined(dst), "N,N,...",
                         [&dst](std::string_view v, std::string& err) {
                             std::vector<size_t> parsed;
                             size_t pos = 0;
                             while (pos <= v.size()) {
                                 const size_t comma = v.find(',', pos);
                                 const std::string_view tok =
                                     v.substr(pos, comma == std::string_view::npos
                                                       ? std::string_view::npos
                                                       : comma - pos);
                                 size_t value = 0;
                                 if (!parseUnsigned(trim(tok), err, value)) return false;
                                 parsed.push_back(value);
                                 if (comma == std::string_view::npos) break;
                                 pos = comma + 1;
                             }
                             dst = std::move(parsed);
                             return true;
                         });
    }

    /// String-to-enum option.  The accepted spellings are recorded in the help
    /// text and repeated in the error message from the same list that does the
    /// matching, so a new enumerator cannot be accepted-but-undocumented.
    template <typename E>
    ArgParser& enumOpt(std::string_view flag, E& dst,
                       std::initializer_list<std::pair<std::string_view, E>> names,
                       std::string_view help) {
        std::vector<std::pair<std::string, E>> table(names.begin(), names.end());
        std::string accepted;
        std::string defaultText = "?";
        for (const auto& [name, value] : table) {
            if (!accepted.empty()) accepted += "|";
            accepted += name;
            if (value == dst) defaultText = name;
        }
        return addValued(flag, std::format("{} [{}]", help, accepted), defaultText, "MODE",
                         [&dst, table = std::move(table), accepted](std::string_view v,
                                                                   std::string& err) {
                             for (const auto& [name, value] : table) {
                                 if (name == v) {
                                     dst = value;
                                     return true;
                                 }
                             }
                             err = std::format("unknown value '{}' (expected one of: {})", v,
                                               accepted);
                             return false;
                         });
    }

    enum class Outcome { Run, Help, Error };

    /// Parses argv[1..].  Help goes to stdout (it was asked for); errors go to
    /// stderr followed by the usage block, so a mistyped flag in a sweep script
    /// is visible without the caller reimplementing the diagnostic.
    Outcome parse(int argc, char** argv) {
        argvEcho_.clear();
        argvEcho_.reserve(static_cast<size_t>(argc < 0 ? 0 : argc));
        for (int i = 0; i < argc; ++i) argvEcho_.emplace_back(argv[i] ? argv[i] : "");

        for (size_t i = 1; i < argvEcho_.size(); ++i) {
            const std::string& arg = argvEcho_[i];
            if (arg == "--help" || arg == "-h") {
                std::cout << usage();
                return Outcome::Help;
            }
            if (arg.size() < 2 || arg[0] != '-') return fail(std::format(
                "unexpected positional argument '{}'", arg));

            // `--flag=value` and `--flag value` are both accepted; the inline form
            // is what shell loops generate, the separated form what humans type.
            std::string        name  = arg;
            std::string_view   inlineValue;
            bool               haveInline = false;
            if (const size_t eq = arg.find('='); eq != std::string::npos) {
                name        = arg.substr(0, eq);
                inlineValue = std::string_view(arg).substr(eq + 1);
                haveInline  = true;
            }

            Entry* e = find(name);
            if (e == nullptr) return fail(std::format("unknown flag '{}'", name));

            std::string_view value;
            if (e->kind == Entry::Kind::Value) {
                if (haveInline) {
                    value = inlineValue;
                } else {
                    if (i + 1 >= argvEcho_.size())
                        return fail(std::format("flag '{}' requires a value", name));
                    value = argvEcho_[++i];
                }
            } else if (haveInline) {
                return fail(std::format("flag '{}' takes no value", name));
            }

            std::string err;
            if (!e->set(value, err)) {
                return fail(err.empty()
                                ? std::format("invalid value '{}' for '{}'", value, name)
                                : std::format("{}: {}", name, err));
            }
        }
        return Outcome::Run;
    }

    /// Full argv including argv[0], for the run manifest.  Populated by parse();
    /// empty before it, which is why the manifest is captured after parsing.
    const std::vector<std::string>& argvEcho() const { return argvEcho_; }

    std::string usage() const {
        std::string out = std::format("{}\n\n{}\n\nUsage: {} [options]\n", programName_,
                                      description_, programName_);
        size_t width = 0;
        for (const auto& e : entries_)
            if (e.kind != Entry::Kind::Group) width = std::max(width, invocation(e).size());

        for (const auto& e : entries_) {
            if (e.kind == Entry::Kind::Group) {
                out += std::format("\n{}\n", e.text);
                continue;
            }
            const std::string inv = invocation(e);
            out += std::format("  {:<{}}  {} (default {})\n", inv, width, e.help,
                               e.defaultText);
        }
        out += std::format("\n  {:<{}}  {}\n", "--help, -h", width, "show this message");
        return out;
    }

  private:
    struct Entry {
        enum class Kind { Group, Flag, Value };
        Kind        kind{Kind::Value};
        std::string flag;
        std::string help;
        std::string defaultText;
        std::string valueName;
        std::string text;  ///< group title
        std::function<bool(std::string_view, std::string&)> set;
    };

    ArgParser& addValued(std::string_view flag, std::string_view help,
                         std::string defaultText, std::string_view valueName,
                         std::function<bool(std::string_view, std::string&)> set) {
        Entry e;
        e.kind        = Entry::Kind::Value;
        e.flag        = normalize(flag);
        e.help        = std::string(help);
        e.defaultText = std::move(defaultText);
        e.valueName   = std::string(valueName);
        e.set         = std::move(set);
        entries_.push_back(std::move(e));
        return *this;
    }

    /// Callers may register either `--n` or `n`; both mean the long flag `--n`.
    static std::string normalize(std::string_view flag) {
        if (flag.starts_with("-")) return std::string(flag);
        return std::format("--{}", flag);
    }

    Entry* find(std::string_view name) {
        for (auto& e : entries_)
            if (e.kind != Entry::Kind::Group && e.flag == name) return &e;
        return nullptr;
    }

    std::string invocation(const Entry& e) const {
        return e.kind == Entry::Kind::Value ? std::format("{} {}", e.flag, e.valueName)
                                            : e.flag;
    }

    Outcome fail(std::string_view message) const {
        std::cerr << std::format("{}: error: {}\n\n", programName_, message) << usage();
        return Outcome::Error;
    }

    static std::string_view trim(std::string_view s) {
        while (!s.empty() && (s.front() == ' ' || s.front() == '\t')) s.remove_prefix(1);
        while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) s.remove_suffix(1);
        return s;
    }

    // Both numeric parsers require the WHOLE token to be consumed.  strtoull on
    // "1e6" or "16k" would otherwise return 1 and leave the driver sweeping a
    // single element, which looks like a fast codec rather than a typo.
    static bool parseUnsigned(std::string_view text, std::string& err, size_t& out) {
        const std::string s(text);
        if (s.empty()) {
            err = "expected an integer, got an empty value";
            return false;
        }
        if (s.front() == '-') {
            err = std::format("expected a non-negative integer, got '{}'", s);
            return false;
        }
        char*             end   = nullptr;
        errno                   = 0;
        const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
        if (end != s.c_str() + s.size() || errno == ERANGE) {
            err = std::format("'{}' is not an integer", s);
            return false;
        }
        out = static_cast<size_t>(v);
        return true;
    }

    static bool parseDouble(std::string_view text, std::string& err, double& out) {
        const std::string s(text);
        if (s.empty()) {
            err = "expected a number, got an empty value";
            return false;
        }
        char* end = nullptr;
        errno     = 0;
        const double v = std::strtod(s.c_str(), &end);
        if (end != s.c_str() + s.size() || errno == ERANGE) {
            err = std::format("'{}' is not a number", s);
            return false;
        }
        out = v;
        return true;
    }

    static std::string joined(const std::vector<std::string>& v) {
        if (v.empty()) return "none";
        std::string out;
        for (const auto& s : v) {
            if (!out.empty()) out += ",";
            out += s;
        }
        return out;
    }

    static std::string joined(const std::vector<size_t>& v) {
        if (v.empty()) return "none";
        std::string out;
        for (size_t x : v) {
            if (!out.empty()) out += ",";
            out += std::format("{}", x);
        }
        return out;
    }

    std::string              programName_;
    std::string              description_;
    std::vector<Entry>       entries_;
    std::vector<std::string> argvEcho_;
};

}  // namespace encodings::benchmark
