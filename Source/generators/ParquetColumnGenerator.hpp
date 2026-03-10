#pragma once

#include "generators/DataGenerator.hpp"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace encodings::generators {

using namespace encodings::datagen;

/**
 * @brief Reads values from a single column of a Parquet file, cycling back
 *        to the start when the column is exhausted.
 *
 * The entire column is loaded into memory once at construction time.
 * Subsequent calls to generate() simply copy slices from that in-memory
 * buffer, wrapping around with modulo indexing.
 *
 * Supported column types: any Arrow numeric type that is castable to T
 * (int8/16/32/64, uint8/16/32/64, float, double).
 *
 * Usage:
 *   ParquetColumnGenerator<int32_t> gen("data.parquet", "l_partkey");
 *   auto values = gen.generate(1'000'000);  // wraps if column < 1M rows
 *
 * @tparam T  The C++ type to expose (e.g. int32_t, int64_t, float, double).
 */
template<typename T>
class ParquetColumnGenerator : public DataGenerator<T> {
public:
    /**
     * @param filePath   Path to the Parquet file.
     * @param columnName Name of the column to read.
     */
    ParquetColumnGenerator(std::filesystem::path filePath, std::string columnName)
        : filePath_(std::move(filePath))
        , columnName_(std::move(columnName))
        , cursor_(0)
    {
        loadColumn();
    }

    // -----------------------------------------------------------------------
    // DataGenerator interface
    // -----------------------------------------------------------------------

    std::vector<T> generate(size_t count) override {
        if (values_.empty()) {
            throw std::runtime_error(
                "ParquetColumnGenerator: column '" + columnName_ + "' is empty");
        }

        std::vector<T> result;
        result.reserve(count);

        for (size_t i = 0; i < count; ++i) {
            result.push_back(values_[cursor_]);
            cursor_ = (cursor_ + 1) % values_.size();
        }

        return result;
    }

    std::string name() const override {
        return "ParquetColumn(" + filePath_.filename().string() + "/" + columnName_ + ")";
    }

    /** Reset the read cursor back to the beginning of the column. */
    void reset() override {
        cursor_ = 0;
    }

    std::map<std::string, std::string> getConfig() const override {
        return {
            {"file",   filePath_.string()},
            {"column", columnName_},
            {"rows",   std::to_string(values_.size())},
        };
    }

    /** Number of rows in the loaded column. */
    size_t columnSize() const { return values_.size(); }

private:
    std::filesystem::path filePath_;
    std::string           columnName_;
    std::vector<T>        values_;
    size_t                cursor_;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    void loadColumn() {
        // Open file
        auto maybeFile = arrow::io::ReadableFile::Open(filePath_.string());
        if (!maybeFile.ok()) {
            throw std::runtime_error(
                "ParquetColumnGenerator: cannot open '" + filePath_.string() +
                "': " + maybeFile.status().ToString());
        }
        std::shared_ptr<arrow::io::ReadableFile> file = *maybeFile;

        // Build reader using the non-deprecated Result-returning overload
        parquet::arrow::FileReaderBuilder builder;
        PARQUET_THROW_NOT_OK(builder.Open(file));
        std::unique_ptr<parquet::arrow::FileReader> reader;
        PARQUET_THROW_NOT_OK(builder.Build(&reader));

        // Resolve column index by name
        auto schema = getSchema(*reader);
        int colIdx   = schema->GetFieldIndex(columnName_);
        if (colIdx < 0) {
            throw std::runtime_error(
                "ParquetColumnGenerator: column '" + columnName_ +
                "' not found in '" + filePath_.string() + "'. " +
                "Available columns: " + listFields(*schema));
        }

        // Read just this one column as a ChunkedArray
        std::shared_ptr<arrow::ChunkedArray> chunked;
        arrow::Status st = reader->ReadColumn(colIdx, &chunked);
        if (!st.ok()) {
            throw std::runtime_error(
                "ParquetColumnGenerator: failed to read column '" +
                columnName_ + "': " + st.ToString());
        }

        // Cast to the target Arrow type and flatten into values_
        auto targetType = arrowTypeFor<T>();
        extractValues(*chunked, targetType);
    }

    std::shared_ptr<arrow::Schema> getSchema(parquet::arrow::FileReader& reader) {
        std::shared_ptr<arrow::Schema> schema;
        arrow::Status st = reader.GetSchema(&schema);
        if (!st.ok()) {
            throw std::runtime_error(
                "ParquetColumnGenerator: cannot read schema: " + st.ToString());
        }
        return schema;
    }

    static std::string listFields(const arrow::Schema& schema) {
        std::string out;
        for (int i = 0; i < schema.num_fields(); ++i) {
            if (i) out += ", ";
            out += schema.field(i)->name();
        }
        return out;
    }

    void extractValues(const arrow::ChunkedArray& chunked,
                       std::shared_ptr<arrow::DataType> targetType)
    {
        values_.reserve(static_cast<size_t>(chunked.length()));

        for (const auto& chunk : chunked.chunks()) {
            // Cast chunk to the desired Arrow numeric type if needed
            std::shared_ptr<arrow::Array> arr = chunk;
            if (!arr->type()->Equals(targetType)) {
                auto maybeArr = arrow::compute::Cast(*arr, targetType);
                if (!maybeArr.ok()) {
                    throw std::runtime_error(
                        "ParquetColumnGenerator: cannot cast column '" +
                        columnName_ + "' from " + arr->type()->ToString() +
                        " to " + targetType->ToString() + ": " +
                        maybeArr.status().ToString());
                }
                arr = maybeArr.ValueOrDie();
            }

            // Downcast to the concrete typed array and copy values
            using ArrowArrayType = typename arrow::CTypeTraits<T>::ArrayType;
            const auto& typed = static_cast<const ArrowArrayType&>(*arr);
            for (int64_t i = 0; i < typed.length(); ++i) {
                if (typed.IsValid(i)) {
                    values_.push_back(static_cast<T>(typed.Value(i)));
                }
                // Null values are silently skipped
            }
        }
    }

    /** Map a C++ type to its corresponding Arrow DataType. */
    template<typename U>
    static std::shared_ptr<arrow::DataType> arrowTypeFor() {
        if constexpr      (std::is_same_v<U, int8_t>)   return arrow::int8();
        else if constexpr (std::is_same_v<U, int16_t>)  return arrow::int16();
        else if constexpr (std::is_same_v<U, int32_t>)  return arrow::int32();
        else if constexpr (std::is_same_v<U, int64_t>)  return arrow::int64();
        else if constexpr (std::is_same_v<U, uint8_t>)  return arrow::uint8();
        else if constexpr (std::is_same_v<U, uint16_t>) return arrow::uint16();
        else if constexpr (std::is_same_v<U, uint32_t>) return arrow::uint32();
        else if constexpr (std::is_same_v<U, uint64_t>) return arrow::uint64();
        else if constexpr (std::is_same_v<U, float>)    return arrow::float32();
        else if constexpr (std::is_same_v<U, double>)   return arrow::float64();
        else static_assert(sizeof(U) == 0,
            "ParquetColumnGenerator: unsupported type T. "
            "Use int8/16/32/64, uint8/16/32/64, float, or double.");
    }
};

} // namespace encodings::generators
