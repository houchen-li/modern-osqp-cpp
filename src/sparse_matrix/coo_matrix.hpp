/**
 * @file coo_matrix.hpp
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

#include <concepts>
#include <unordered_map>
#include <vector>

#include "spdlog/spdlog.h"

#include "sparse_matrix/sparse_matrix.hpp"

namespace osqp {

template <std::floating_point Scalar, std::integral Index>
class LilMatrix;

template <std::floating_point Scalar, std::integral Index>
class CscMatrix;

template <std::floating_point Scalar, std::integral Index>
class CsrMatrix;

template <std::floating_point Scalar = double, std::integral Index = int>
class [[nodiscard]] CooMatrix final : public SparseMatrix<CooMatrix<Scalar, Index>, Scalar, Index> {
    friend class LilMatrix<Scalar, Index>;
    friend class CscMatrix<Scalar, Index>;
    friend class CsrMatrix<Scalar, Index>;

  public:
    [[using gnu: always_inline]] explicit CooMatrix(std::size_t nrows, std::size_t ncols) noexcept
        : nrows_{nrows}, ncols_{ncols} {}

    CooMatrix() noexcept = default;
    CooMatrix(const CooMatrix& other) noexcept = default;
    CooMatrix(CooMatrix&& other) noexcept = default;
    auto operator=(const CooMatrix& other) noexcept -> CooMatrix& = default;
    auto operator=(CooMatrix&& other) noexcept -> CooMatrix& = default;
    ~CooMatrix() noexcept override = default;

    [[using gnu: flatten, leaf]] CooMatrix(const LilMatrix<Scalar, Index>& lil_matrix) noexcept
        : nrows_{lil_matrix.nrows_}, ncols_{lil_matrix.ncols_} {
        const std::size_t values_size = lil_matrix.nnzs();
        offsets_map_.reserve(values_size);
        values_.reserve(values_size);
        row_indices_.reserve(values_size);
        col_indices_.reserve(values_size);
        for (const auto& [row, offsets_map] : lil_matrix.row_offsets_map_) {
            for (const auto& [col, offset] : offsets_map) {
                offsets_map_.emplace(IndexPair{row, col}, values_.size());
                values_.push_back(lil_matrix.row_values_map_.at(row)[offset]);
                row_indices_.push_back(row);
                col_indices_.push_back(col);
            }
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
        return values_.size();
    }

    [[using gnu: flatten, leaf]]
    auto resize(std::size_t nrows, std::size_t ncols) noexcept -> void {
        if (nrows < nrows_ || ncols < ncols_) {
            const std::size_t values_size = values_.size();
            std::size_t j{0};
            for (std::size_t i{0}; i < values_size; ++i) {
                if (row_indices_[i] >= static_cast<Index>(nrows) ||
                    col_indices_[i] >= static_cast<Index>(ncols)) {
                    offsets_map_.erase(IndexPair{row_indices_[i], col_indices_[i]});
                    continue;
                }
                if (j != i) {
                    offsets_map_[IndexPair{row_indices_[i], col_indices_[i]}] = j;
                    values_[j] = values_[i];
                    row_indices_[j] = row_indices_[i];
                    col_indices_[j] = col_indices_[i];
                }
                ++j;
            }
            values_.resize(j);
            row_indices_.resize(j);
            col_indices_.resize(j);
            values_.shrink_to_fit();
            row_indices_.shrink_to_fit();
            col_indices_.shrink_to_fit();
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
        IndexPair index_pair{row, col};
        if (const auto search = offsets_map_.find(index_pair); search != offsets_map_.end()) {
            return values_[search->second];
        }
        return 0.0;
    }

    [[using gnu: always_inline, hot]]
    auto updateCoeff(Index row, Index col, Scalar coeff) noexcept -> void {
        if (row >= static_cast<Index>(nrows_) || col >= static_cast<Index>(ncols_)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" and \"col\" should be less than \"nrows\" "
                "and \"ncols\" respectively: (row, col) = ({0:d}, {1:d}) while (nrows, ncols) = "
                "({2:d}, {3:d}).",
                row, col, nrows_, ncols_
            );
            return;
        }
        IndexPair index_pair{row, col};
        if (auto search = offsets_map_.find(index_pair); search != offsets_map_.end()) {
            const std::size_t offset = search->second;
            values_[offset] = coeff;
        } else {
            if (coeff != 0.0) {
                offsets_map_[index_pair] = values_.size();
                values_.push_back(coeff);
                row_indices_.push_back(row);
                col_indices_.push_back(col);
            }
        }
        return;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto values() const noexcept -> const std::vector<Scalar>& {
        return values_;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto rowIndices() const noexcept -> const std::vector<Index>& {
        return row_indices_;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto colIndices() const noexcept -> const std::vector<Index>& {
        return col_indices_;
    }

    [[using gnu: always_inline]]
    auto clear() noexcept -> void {
        offsets_map_.clear();
        values_.clear();
        row_indices_.clear();
        col_indices_.clear();
        return;
    }

  private:
    struct [[nodiscard]] IndexPair final {
        Index row;
        Index col;

        [[using gnu: pure, always_inline]]
        auto
        operator==(IndexPair index_pair) const noexcept -> bool {
            return row == index_pair.row && col == index_pair.col;
        }
    };

    struct [[nodiscard]] IndexPairHash final {
        [[using gnu: pure, always_inline]]
        auto
        operator()(typename CooMatrix<Scalar, Index>::IndexPair index_pair) const noexcept
            -> std::size_t {
            return std::hash<Index>()(index_pair.row) ^ std::hash<Index>()(index_pair.col);
        }
    };

    struct [[nodiscard]] IndexPairRowMajorCompare final {
        [[using gnu: pure, always_inline, leaf]]
        auto
        operator()(const IndexPair& lhs, const IndexPair& rhs) const noexcept -> bool {
            if (lhs.row < rhs.row) {
                return true;
            } else if (lhs.row > rhs.row) {
                return false;
            }
            if (lhs.col < rhs.col) {
                return true;
            } else if (lhs.col > rhs.col) {
                return false;
            }
            return false;
        }
    };

    struct [[nodiscard]] IndexPairColumnMajorCompare final {
        [[using gnu: pure, always_inline, leaf]]
        auto
        operator()(const IndexPair& lhs, const IndexPair& rhs) const noexcept -> bool {
            if (lhs.col < rhs.col) {
                return true;
            } else if (lhs.col > rhs.col) {
                return false;
            }
            if (lhs.row < rhs.row) {
                return true;
            } else if (lhs.row > rhs.row) {
                return false;
            }
            return false;
        }
    };

    std::size_t nrows_{0};
    std::size_t ncols_{0};
    std::unordered_map<
        typename CooMatrix<Scalar, Index>::IndexPair, std::size_t,
        typename CooMatrix<Scalar, Index>::IndexPairHash>
        offsets_map_{};
    std::vector<Scalar> values_{};
    std::vector<Index> row_indices_{};
    std::vector<Index> col_indices_{};
};

} // namespace osqp
