#pragma once

// Typed, name-addressed result output for the benchmark drivers.
//
// Two invariants drive the whole design.
//
// 1. Fields are set BY NAME and an unknown name throws.  The emitter this
//    replaces wrote a hand-counted `",,,,,,"` for a skipped cell, so inserting a
//    column silently shifted every value in every row after it — a corruption
//    that survives into the plots and is invisible in the file.
//
// 2. There is ONE typed source of truth: an Arrow schema plus its
//    ArrayBuilders.  Parquet is written from it by parquet::arrow, and CSV is
//    serialised from the very same finished Arrow arrays by the small serialiser
//    below.  This is not a preference: this Arrow build sets ARROW_CSV=OFF (see
//    Source/generators/CMakeLists.txt) so arrow/csv/writer.h does not exist.
//    Sharing the arrays rather than the intent means column order and null
//    rendering cannot diverge between the two formats, which is the only reason
//    a CSV run and a Parquet run of the same sweep are comparable.

// Arrow's own headers are not clean under this project's -Wall -Wextra
// -Wpedantic -Werror (unused parameters in builder_run_end.h, a deprecated
// NewRowGroup overload in parquet/arrow/writer.h).  The CMake build gets away
// with it because the arrow include dir arrives through an IMPORTED target and
// is therefore treated as -isystem; suppressing here as well keeps a plain
// `g++ -I build/arrow-install/include` compile of this header working too.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#pragma GCC diagnostic pop

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace encodings::benchmark {

enum class ResultFormat { Csv, Parquet };

struct ColumnSpec {
    std::string                        name;
    std::shared_ptr<arrow::DataType>   type;
};

/// Convenience constructors for the four column types the drivers use.  Kept as
/// free functions so a column list reads as a table in the driver source.
inline ColumnSpec doubleCol(std::string name) { return {std::move(name), arrow::float64()}; }
inline ColumnSpec intCol(std::string name)    { return {std::move(name), arrow::int64()}; }
inline ColumnSpec stringCol(std::string name) { return {std::move(name), arrow::utf8()}; }
inline ColumnSpec boolCol(std::string name)   { return {std::move(name), arrow::boolean()}; }

class ResultWriter {
  public:
    ResultWriter(std::filesystem::path path, std::vector<ColumnSpec> columns,
                 ResultFormat format)
        : path_(std::move(path)), columns_(std::move(columns)), format_(format) {
        if (columns_.empty()) throw std::runtime_error("ResultWriter: no columns declared");

        std::vector<std::shared_ptr<arrow::Field>> fields;
        fields.reserve(columns_.size());
        for (size_t i = 0; i < columns_.size(); ++i) {
            const auto& c = columns_[i];
            if (!c.type) throw std::runtime_error(
                std::format("ResultWriter: column '{}' has no type", c.name));
            if (!index_.emplace(c.name, i).second)
                throw std::runtime_error(
                    std::format("ResultWriter: duplicate column '{}'", c.name));
            // Nullable throughout, deliberately: "not applicable" is a null, and a
            // non-nullable schema would force the sentinel numbers this class exists
            // to abolish.
            fields.push_back(arrow::field(c.name, c.type, /*nullable=*/true));
        }
        schema_ = arrow::schema(std::move(fields));

        if (path_.has_parent_path())
            std::filesystem::create_directories(path_.parent_path());

        resetBuilders();

        if (format_ == ResultFormat::Csv) {
            csv_.open(path_, std::ios::out | std::ios::trunc);
            if (!csv_) throw std::runtime_error(
                std::format("ResultWriter: cannot open '{}'", path_.string()));
            for (size_t i = 0; i < columns_.size(); ++i) {
                if (i != 0) csv_ << ',';
                csv_ << csvField(columns_[i].name);
            }
            csv_ << '\n';
            csv_.flush();
        } else {
            auto sink = arrow::io::FileOutputStream::Open(path_.string());
            check(sink.status(), "opening output file");
            auto writer = parquet::arrow::FileWriter::Open(
                *schema_, arrow::default_memory_pool(), *sink);
            check(writer.status(), "opening parquet writer");
            parquet_ = std::move(*writer);
        }
    }

    ResultWriter(const ResultWriter&)            = delete;
    ResultWriter& operator=(const ResultWriter&) = delete;

    /// Finalises on destruction so an early `return` from a driver still leaves a
    /// well-formed file.  Errors are swallowed here — throwing from a destructor
    /// during stack unwinding would replace the real failure with this one — so a
    /// driver that cares calls close() explicitly and gets the exception.
    ~ResultWriter() {
        try {
            close();
        } catch (...) {  // NOLINT: see above
        }
    }

    class Row {
      public:
        Row& set(std::string_view col, double v)          { return put(col, Value{v}); }
        Row& set(std::string_view col, int64_t v)         { return put(col, Value{v}); }
        Row& set(std::string_view col, size_t v)          { return put(col, Value{static_cast<uint64_t>(v)}); }
        Row& set(std::string_view col, std::string_view v){ return put(col, Value{std::string(v)}); }
        Row& set(std::string_view col, bool v)            { return put(col, Value{v}); }

        /// `const char*` needs its own overload: pointer-to-bool is a standard
        /// conversion and would outrank the user-defined one to string_view, so a
        /// literal would land in the bool overload and write `1`.
        Row& set(std::string_view col, const char* v) {
            return put(col, Value{std::string(v == nullptr ? "" : v)});
        }

        /// Any other integer width (`int` literals, `uint32_t` counters).  Without
        /// this, `set(col, 1)` is ambiguous between the double, int64, size_t and
        /// bool overloads; as an exact match this template wins outright.
        template <typename T>
            requires std::integral<T> && (!std::same_as<T, bool>) &&
                     (!std::same_as<T, int64_t>) && (!std::same_as<T, size_t>)
        Row& set(std::string_view col, T v) {
            if constexpr (std::is_signed_v<T>) return put(col, Value{static_cast<int64_t>(v)});
            else return put(col, Value{static_cast<uint64_t>(v)});
        }

        /// Explicit null.  Distinct from "never set" only in intent; both produce a
        /// typed null, and saying it out loud documents a deliberate omission.
        Row& setNull(std::string_view col) {
            owner_->columnIndex(col);  // still validated: a typo must not pass silently
            return *this;
        }

        /// `setIf(cond, col, v)` is the whole "not applicable" idiom: false means
        /// null, never 0 and never -1.
        template <typename T>
        Row& setIf(bool cond, std::string_view col, T v) {
            return cond ? set(col, v) : setNull(col);
        }

      private:
        friend class ResultWriter;
        using Value = std::variant<double, int64_t, uint64_t, std::string, bool>;

        explicit Row(ResultWriter* owner) : owner_(owner), cells_(owner->columns_.size()) {}

        Row& put(std::string_view col, Value v) {
            const size_t i = owner_->columnIndex(col);
            owner_->checkType(i, v);
            cells_[i] = std::move(v);
            return *this;
        }

        ResultWriter*                     owner_;
        std::vector<std::optional<Value>> cells_;
    };

    Row row() { return Row(this); }

    void write(Row&& r) {
        if (r.owner_ != this)
            throw std::runtime_error("ResultWriter: row belongs to a different writer");
        for (size_t i = 0; i < columns_.size(); ++i) append(i, r.cells_[i]);
        ++pending_;
        // Bound the staging cost of a long sweep without the caller having to think
        // about it.  Drivers additionally flush per encoder (see CONVENTIONS.md 6),
        // which is about inspectability rather than memory.
        if (pending_ >= kAutoDrainRows) flush();
    }

    /// CSV: appends the staged rows and flushes the stream, so the partial file is
    /// a valid CSV that a plotting script can read while the sweep is still
    /// running.  Parquet: closes the current row group; the file only becomes
    /// readable at close(), which is inherent to the format's trailing footer.
    void flush() {
        drain();
        if (format_ == ResultFormat::Csv) csv_.flush();
    }

    void close() {
        if (closed_) return;
        drain();
        closed_ = true;  // set before the fallible work so a throwing close is not retried
        if (format_ == ResultFormat::Csv) {
            csv_.flush();
            csv_.close();
        } else if (parquet_) {
            check(parquet_->Close(), "closing parquet writer");
            parquet_.reset();
        }
    }

    const std::shared_ptr<arrow::Schema>& schema() const { return schema_; }
    size_t rowsWritten() const { return rowsWritten_; }

  private:
    static constexpr size_t kAutoDrainRows = 4096;

    using Value = Row::Value;

    static void check(const arrow::Status& s, std::string_view what) {
        if (!s.ok())
            throw std::runtime_error(std::format("ResultWriter: {}: {}", what, s.ToString()));
    }

    size_t columnIndex(std::string_view col) const {
        const auto it = index_.find(std::string(col));
        if (it == index_.end())
            throw std::runtime_error(
                std::format("ResultWriter: no such column '{}'", col));
        return it->second;
    }

    /// Rejects rather than converts.  A double landing in an int64 column would
    /// silently truncate, and a nanosecond count quietly rounded to a whole
    /// microsecond is a wrong number that still looks plausible.
    void checkType(size_t i, const Value& v) const {
        const arrow::Type::type id = columns_[i].type->id();
        const bool ok = std::visit(
            [&](const auto& held) {
                using T = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<T, double>) return id == arrow::Type::DOUBLE;
                else if constexpr (std::is_same_v<T, std::string>) return id == arrow::Type::STRING;
                else if constexpr (std::is_same_v<T, bool>)
                    // Booleans are also accepted into an int64 column, which is how
                    // `skipped`/`truncated` end up as the 0/1 the plots expect
                    // rather than "true"/"false".
                    return id == arrow::Type::BOOL || id == arrow::Type::INT64;
                else return id == arrow::Type::INT64 || id == arrow::Type::UINT64;
            },
            v);
        if (!ok)
            throw std::runtime_error(std::format(
                "ResultWriter: column '{}' is {}, cannot accept a {} value",
                columns_[i].name, columns_[i].type->ToString(), valueTypeName(v)));
    }

    static std::string_view valueTypeName(const Value& v) {
        return std::visit(
            [](const auto& held) -> std::string_view {
                using T = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<T, double>) return "double";
                else if constexpr (std::is_same_v<T, std::string>) return "string";
                else if constexpr (std::is_same_v<T, bool>) return "bool";
                else return "integer";
            },
            v);
    }

    void resetBuilders() {
        builders_.clear();
        builders_.reserve(columns_.size());
        for (const auto& c : columns_) {
            std::unique_ptr<arrow::ArrayBuilder> b;
            check(arrow::MakeBuilder(arrow::default_memory_pool(), c.type, &b),
                  std::format("creating a builder for column '{}'", c.name));
            builders_.push_back(std::move(b));
        }
        pending_ = 0;
    }

    void append(size_t i, const std::optional<Value>& cell) {
        arrow::ArrayBuilder& b = *builders_[i];
        if (!cell) {
            check(b.AppendNull(), "appending a null");
            return;
        }
        const arrow::Type::type id = columns_[i].type->id();
        std::visit(
            [&](const auto& held) {
                using T = std::decay_t<decltype(held)>;
                if constexpr (std::is_same_v<T, double>) {
                    check(static_cast<arrow::DoubleBuilder&>(b).Append(held), "appending");
                } else if constexpr (std::is_same_v<T, std::string>) {
                    check(static_cast<arrow::StringBuilder&>(b).Append(held), "appending");
                } else if constexpr (std::is_same_v<T, bool>) {
                    if (id == arrow::Type::BOOL)
                        check(static_cast<arrow::BooleanBuilder&>(b).Append(held), "appending");
                    else
                        check(static_cast<arrow::Int64Builder&>(b).Append(held ? 1 : 0),
                              "appending");
                } else {
                    if (id == arrow::Type::UINT64)
                        check(static_cast<arrow::UInt64Builder&>(b).Append(
                                  static_cast<uint64_t>(held)), "appending");
                    else
                        check(static_cast<arrow::Int64Builder&>(b).Append(
                                  static_cast<int64_t>(held)), "appending");
                }
            },
            *cell);
    }

    void drain() {
        if (pending_ == 0) return;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        arrays.reserve(builders_.size());
        for (size_t i = 0; i < builders_.size(); ++i) {
            std::shared_ptr<arrow::Array> a;
            check(builders_[i]->Finish(&a),
                  std::format("finishing column '{}'", columns_[i].name));
            arrays.push_back(std::move(a));
        }
        const int64_t rows = static_cast<int64_t>(pending_);
        rowsWritten_ += pending_;
        resetBuilders();

        if (format_ == ResultFormat::Csv) {
            writeCsvChunk(arrays, rows);
        } else {
            const auto table = arrow::Table::Make(schema_, arrays, rows);
            check(parquet_->WriteTable(*table, rows), "writing a row group");
        }
    }

    /// The CSV counterpart of parquet::arrow — same arrays, same order, nulls as
    /// empty fields.
    void writeCsvChunk(const std::vector<std::shared_ptr<arrow::Array>>& arrays,
                       int64_t rows) {
        for (int64_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < arrays.size(); ++c) {
                if (c != 0) csv_ << ',';
                const arrow::Array& a = *arrays[c];
                if (a.IsNull(r)) continue;  // null renders as an empty field
                switch (a.type_id()) {
                    case arrow::Type::DOUBLE:
                        // Shortest round-tripping form.  A %.6g default would collapse
                        // distinct timings into equal strings and make a ratio computed
                        // downstream disagree with the one computed here.
                        csv_ << std::format(
                            "{}", static_cast<const arrow::DoubleArray&>(a).Value(r));
                        break;
                    case arrow::Type::INT64:
                        csv_ << static_cast<const arrow::Int64Array&>(a).Value(r);
                        break;
                    case arrow::Type::UINT64:
                        csv_ << static_cast<const arrow::UInt64Array&>(a).Value(r);
                        break;
                    case arrow::Type::BOOL:
                        csv_ << (static_cast<const arrow::BooleanArray&>(a).Value(r) ? '1'
                                                                                    : '0');
                        break;
                    case arrow::Type::STRING:
                        csv_ << csvField(static_cast<const arrow::StringArray&>(a).GetView(r));
                        break;
                    default:
                        throw std::runtime_error(std::format(
                            "ResultWriter: column '{}' has type {}, which the CSV "
                            "serialiser does not handle",
                            columns_[c].name, a.type()->ToString()));
                }
            }
            csv_ << '\n';
        }
    }

    static std::string csvField(std::string_view s) {
        if (s.find_first_of(",\"\n\r") == std::string_view::npos) return std::string(s);
        std::string out = "\"";
        for (char ch : s) {
            if (ch == '"') out += '"';  // RFC 4180: an embedded quote is doubled
            out += ch;
        }
        out += '"';
        return out;
    }

    std::filesystem::path                             path_;
    std::vector<ColumnSpec>                           columns_;
    ResultFormat                                      format_;
    std::unordered_map<std::string, size_t>           index_;
    std::shared_ptr<arrow::Schema>                    schema_;
    std::vector<std::unique_ptr<arrow::ArrayBuilder>> builders_;
    std::ofstream                                     csv_;
    std::unique_ptr<parquet::arrow::FileWriter>       parquet_;
    size_t                                            pending_{0};
    size_t                                            rowsWritten_{0};
    bool                                              closed_{false};
};

}  // namespace encodings::benchmark
