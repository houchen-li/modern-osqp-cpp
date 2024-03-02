/**
 * @file lil_matrix.hpp
 * @author Houchen Li (houchen_li@hotmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-20
 *
 * @copyright Copyright (c) 2023 Houchen Li
 *            All rights reserved.
 *
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <unordered_map>
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"

#include "sparse_matrix/sparse_matrix.hpp"

namespace osqp {

template <std::floating_point Scalar, std::integral Index>
class CooMatrix;

template <std::floating_point Scalar, std::integral Index>
class CscMatrix;

template <std::floating_point Scalar = double, std::integral Index = int>
class [[nodiscard]] LilMatrix final : public SparseMatrix<LilMatrix<Scalar, Index>, Scalar, Index> {
    friend class CooMatrix<Scalar, Index>;

  public:
    [[using gnu: always_inline]] explicit LilMatrix(std::size_t nrows, std::size_t ncols) noexcept
        : nrows_{nrows}, ncols_{ncols} {}

    LilMatrix() noexcept = default;
    LilMatrix(const LilMatrix& other) noexcept = default;
    LilMatrix(LilMatrix&& other) noexcept = default;
    auto operator=(const LilMatrix& other) noexcept -> LilMatrix& = default;
    auto operator=(LilMatrix&& other) noexcept -> LilMatrix& = default;
    ~LilMatrix() noexcept override = default;

    [[using gnu: flatten, leaf]] LilMatrix(const CooMatrix<Scalar, Index>& coo_matrix) noexcept
        : nrows_{coo_matrix.nrows_}, ncols_{coo_matrix.ncols_}, nnzs_{coo_matrix.values_.size()} {
        for (const auto& [index_pair, offset] : coo_matrix.offsets_map_) {
            row_offsets_map_[index_pair.row].emplace(
                index_pair.col, row_values_map_[index_pair.row].size()
            );
            row_values_map_[index_pair.row].emplace_back(coo_matrix.values_[offset]);
            row_indices_map_[index_pair.row].emplace_back(index_pair.col);
        }
        for (auto& [row, values] : row_values_map_) {
            values.shrink_to_fit();
        }
        for (auto& [row, indices] : row_indices_map_) {
            indices.shrink_to_fit();
        }
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto nrows() const noexcept -> std::size_t {
        return nrows_;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto ncols() const noexcept -> std::size_t {
        return ncols_;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto nnzs() const noexcept -> std::size_t {
        return nnzs_;
    }

    [[using gnu: flatten, leaf]]
    auto resize(std::size_t nrows, std::size_t ncols) noexcept -> void {
        if (nrows < nrows_ || ncols < ncols_) {
            std::vector<Index> remove_rows;
            for (const auto& [row, row_offsets_map] : row_offsets_map_) {
                if (row >= static_cast<Index>(nrows)) {
                    remove_rows.push_back(row);
                }
            }
            for (const Index row : remove_rows) {
                row_offsets_map_.erase(row);
                nnzs_ -= row_values_map_[row].size();
                row_values_map_.erase(row);
                row_indices_map_.erase(row);
            }
            remove_rows.clear();
            for (auto& [row, indices] : row_indices_map_) {
                nnzs_ -= indices.size();
                std::size_t j{0};
                for (std::size_t i{0}; i < indices.size(); ++i) {
                    if (indices[i] >= static_cast<Index>(ncols)) {
                        row_offsets_map_[row].erase(indices[i]);
                        continue;
                    }
                    if (j != i) {
                        row_offsets_map_[row][indices[i]] = j;
                        row_values_map_[row][j] = row_values_map_[row][i];
                        indices[j] = indices[i];
                    }
                    ++j;
                }
                if (j != 0) {
                    row_values_map_[row].resize(j);
                    row_indices_map_[row].resize(j);
                    indices.shrink_to_fit();
                    indices.shrink_to_fit();
                    nnzs_ += j;
                } else {
                    remove_rows.push_back(row);
                }
            }
            for (const Index row : remove_rows) {
                row_offsets_map_.erase(row);
                row_values_map_.erase(row);
                row_indices_map_.erase(row);
            }
            remove_rows.clear();
        }
        nrows_ = nrows;
        ncols_ = ncols;
        return;
    }

    [[using gnu: pure, always_inline, hot]]
    auto coeff(Index row, Index col) const noexcept -> Scalar {
        if (row >= static_cast<Index>(nrows_) || col >= static_cast<Index>(ncols_)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" and \"col\" should be less than \"nrows\" "
                "and \"ncols\" respectively: (row, col) = ({0:d}, {1:d}) while (nrows, ncols) = "
                "({2:d}, {3:d}).",
                row, col, nrows_, ncols_
            );
            return 0.0;
        }
        if (const auto search_row = row_offsets_map_.find(row);
            search_row != row_offsets_map_.end()) {
            if (const auto search_col = search_row->second.find(col);
                search_col != search_row->second.end()) {
                const std::size_t offset = search_col->second;
                return row_values_map_.at(row)[offset];
            }
        }
        return 0.0;
    }

    [[using gnu: hot]]
    auto updateRow(Index row, std::vector<Scalar> values, std::vector<Index> indices) noexcept
        -> void {
        if (row >= static_cast<Index>(nrows_)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" should be less than \"nrows\": row = {0:d} "
                "while nrows = {1:d}.",
                row, nrows_
            );
            return;
        }
        if (values.size() != indices.size()) {
            const std::size_t values_size{std::min(values.size(), indices.size())};
            values.resize(values_size);
            indices.resize(values_size);
        }
        if (auto search = row_offsets_map_.find(row); search != row_offsets_map_.end()) {
            row_offsets_map_.erase(search);
        }
        if (auto search = row_values_map_.find(row); search != row_values_map_.end()) {
            nnzs_ -= search->second.size();
            row_values_map_.erase(search);
        }
        if (auto search = row_indices_map_.find(row); search != row_indices_map_.end()) {
            row_indices_map_.erase(search);
        }
        if (values.size() == 0) {
            return;
        }
        const std::size_t values_size = values.size();
        std::unordered_map<Index, std::size_t> indices_map;
        indices_map.reserve(values_size);
        std::size_t j = 0;
        for (std::size_t i = 0; i < values_size; ++i) {
            if (indices[i] >= static_cast<Index>(ncols_)) {
                spdlog::warn(
                    "Out of range issue detected! \"col\" should be less than \"ncols\": col = "
                    "{0:d} while ncols = {1:d}.",
                    indices[i], ncols_
                );
                continue;
            }
            if (values[i] == 0.0) {
                continue;
            }
            indices_map.emplace(indices[i], j);
            if (j != i) {
                values[j] = values[i];
                indices[j] = indices[i];
            }
            ++j;
        }
        if (j != 0) {
            values.resize(j);
            indices.resize(j);
            values.shrink_to_fit();
            indices.shrink_to_fit();
            row_offsets_map_.emplace(row, std::move(indices_map));
            nnzs_ += values.size();
            row_values_map_.emplace(row, std::move(values));
            row_indices_map_.emplace(row, std::move(indices));
        }
        return;
    }

    [[using gnu: hot]]
    auto updateRow(Index row, const std::unordered_map<Index, Scalar>& values_map) noexcept
        -> void {
        if (row >= static_cast<Index>(nrows_)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" should be less than \"nrows\": row = {0:d} "
                "while nrows = {1:d}.",
                row, nrows_
            );
            return;
        }
        if (auto search = row_offsets_map_.find(row); search != row_offsets_map_.end()) {
            row_offsets_map_.erase(search);
        }
        if (auto search = row_values_map_.find(row); search != row_values_map_.end()) {
            nnzs_ -= search->second.size();
            row_values_map_.erase(search);
        }
        if (auto search = row_indices_map_.find(row); search != row_indices_map_.end()) {
            row_indices_map_.erase(search);
        }
        if (values_map.size() == 0) {
            return;
        }
        std::unordered_map<Index, std::size_t> indices_map;
        indices_map.reserve(values_map.size());
        std::vector<Scalar> values;
        values.reserve(values_map.size());
        std::vector<Index> indices;
        indices.reserve(values_map.size());
        for (const auto& [col, value] : values_map) {
            if (col >= static_cast<Index>(ncols_)) {
                spdlog::warn(
                    "Out of range issue detected! \"col\" should be less than \"ncols\": col = "
                    "{0:d} while ncols = {1:d}.",
                    col, ncols_
                );
                continue;
            }
            if (value == 0.0) {
                continue;
            }
            indices_map.emplace(col, values.size());
            values.push_back(value);
            indices.push_back(col);
        }
        if (!values.empty()) {
            values.shrink_to_fit();
            indices.shrink_to_fit();
            row_offsets_map_.emplace(row, std::move(indices_map));
            nnzs_ += values.size();
            row_values_map_.emplace(row, std::move(values));
            row_indices_map_.emplace(row, std::move(indices));
        }
        return;
    }

    [[using gnu: pure, always_inline]]
    auto rowValues(Index row) const noexcept -> std::vector<Scalar> {
        if (row >= static_cast<Index>(nrows_)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" should be less than \"nrows\": row = {0:d} "
                "while nrows = {1:d}.",
                row, nrows_
            );
            return {};
        }
        if (const auto search = row_offsets_map_.find(row); search != row_offsets_map_.end()) {
            return row_values_map_.at(row);
        }
        return {};
    }

    [[using gnu: pure, always_inline]]
    auto rowIndices(Index row) const noexcept -> std::vector<Index> {
        if (row >= static_cast<Index>(nrows_)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" should be less than \"nrows\": row = {0:d} "
                "while nrows = {1:d}.",
                row, nrows_
            );
            return {};
        }
        if (const auto search = row_offsets_map_.find(row); search != row_offsets_map_.end()) {
            return row_indices_map_.at(row);
        }
        return {};
    }

    [[using gnu: always_inline]]
    auto clear() noexcept -> void {
        nnzs_ = 0;
        row_offsets_map_.clear();
        row_values_map_.clear();
        row_indices_map_.clear();
        return;
    }

  private:
    std::size_t nrows_{0};
    std::size_t ncols_{0};
    std::size_t nnzs_{0};
    std::unordered_map<Index, std::unordered_map<Index, std::size_t>> row_offsets_map_{};
    std::unordered_map<Index, std::vector<Scalar>> row_values_map_{};
    std::unordered_map<Index, std::vector<Index>> row_indices_map_{};
};

} // namespace osqp
