/**
 * @file csc_matrix.hpp
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
#include <map>
#include <utility>
#include <vector>

#include "spdlog/spdlog.h"

#include "sparse_matrix/sparse_matrix.hpp"

namespace osqp {

template <std::floating_point Scalar, std::integral Index>
class CooMatrix;

template <std::floating_point Scalar, std::integral Index>
class LilMatrix;

template <std::floating_point Scalar = double, std::integral Index = int>
class [[nodiscard]] CscMatrix final : public SparseMatrix<CscMatrix<Scalar, Index>, Scalar, Index> {
  public:
    [[using gnu: always_inline]] explicit CscMatrix(std::size_t nrows, std::size_t ncols) noexcept
        : nrows_{nrows}, outer_indices_(ncols + 1, 0) {}

    CscMatrix() noexcept = default;
    CscMatrix(const CscMatrix& other) noexcept = default;
    CscMatrix(CscMatrix&& other) noexcept = default;
    auto operator=(const CscMatrix& other) noexcept -> CscMatrix& = default;
    auto operator=(CscMatrix&& other) noexcept -> CscMatrix& = default;
    ~CscMatrix() noexcept override = default;

    [[using gnu: flatten, leaf]] CscMatrix(const CooMatrix<Scalar, Index>& coo_matrix) noexcept
        : nrows_{coo_matrix.nrows_} {
        std::map<
            typename CooMatrix<Scalar, Index>::IndexPair, std::size_t,
            typename CooMatrix<Scalar, Index>::IndexPairColumnMajorCompare>
            offsets_map{coo_matrix.offsets_map_.cbegin(), coo_matrix.offsets_map_.cend()};
        const std::size_t nnzs = coo_matrix.nnzs();
        values_.reserve(nnzs);
        inner_indices_.reserve(nnzs);
        outer_indices_.reserve(coo_matrix.ncols_ + 1);
        outer_indices_.push_back(0);
        for (const auto& [index_pair, offset] : offsets_map) {
            values_.push_back(coo_matrix.values_[offset]);
            inner_indices_.push_back(index_pair.row);
            const std::size_t diff_col = index_pair.col - outer_indices_.size() + 1;
            for (std::size_t i{0}; i < diff_col; ++i) {
                outer_indices_.push_back(values_.size() - 1);
            }
        }
        for (std::size_t i = outer_indices_.size(); i < coo_matrix.ncols_ + 1; ++i) {
            outer_indices_.push_back(nnzs);
        }
    }

    [[using gnu: always_inline]] CscMatrix(const LilMatrix<Scalar, Index>& lil_matrix) noexcept {
        CooMatrix<Scalar, Index> coo_matrix{lil_matrix};
        *this = std::move(coo_matrix);
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto nrows() const noexcept -> std::size_t {
        return nrows_;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto ncols() const noexcept -> std::size_t {
        return outer_indices_.size() - 1;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto nnzs() const noexcept -> std::size_t {
        return values_.size();
    }

    [[using gnu: flatten, leaf]]
    auto resize(std::size_t nrows, std::size_t ncols) noexcept -> void {
        if (ncols >= outer_indices_.size() - 1) {
            for (std::size_t i = outer_indices_.size() - 1; i < ncols; ++i) {
                outer_indices_.push_back(outer_indices_.back());
            }
        } else {
            outer_indices_.resize(ncols + 1);
            values_.resize(outer_indices_.back());
            inner_indices_.resize(outer_indices_.back());
        }
        const std::size_t values_size{values_.size()};
        if (nrows < nrows_) {
            std::size_t j{0}, k{0};
            nrows_ = nrows;
            for (std::size_t i{0}; i < values_size; ++i) {
                if (inner_indices_[i] >= static_cast<Index>(nrows_)) {
                    continue;
                }
                if (j != i) {
                    values_[j] = values_[i];
                    inner_indices_[j] = inner_indices_[i];
                    if (i == static_cast<std::size_t>(outer_indices_[k])) {
                        outer_indices_[k] = j;
                    }
                }
                ++j;
                if (i == static_cast<std::size_t>(outer_indices_[k])) {
                    ++k;
                }
            }
            outer_indices_[ncols] = j;
            values_.resize(j);
            inner_indices_.resize(j);
            outer_indices_.shrink_to_fit();
            values_.shrink_to_fit();
            inner_indices_.shrink_to_fit();
        }
        return;
    }

    [[using gnu: pure, always_inline, hot]]
    auto coeff(Index row, Index col) const noexcept -> Scalar {
        if (row >= static_cast<Index>(nrows_) ||
            col >= static_cast<Index>(outer_indices_.size() - 1)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" and \"col\" should be less than \"nrows\" "
                "and \"ncols\" respectively: (row, col) = ({0:d}, {1:d}) while (nrows, ncols) = "
                "({2:d}, {3:d}).",
                row, col, nrows_, outer_indices_.size() - 1
            );
            return 0.0;
        }
        if (const auto search = std::find(
                inner_indices_.cbegin() + outer_indices_[col],
                inner_indices_.cbegin() + outer_indices_[col + 1], row
            );
            search != inner_indices_.cbegin() + outer_indices_[col + 1]) {
            return values_[search - inner_indices_.cbegin()];
        }
        return 0.0;
    }

    [[using gnu: hot]]
    auto updateCoeff(Index row, Index col, Scalar value) noexcept -> void {
        if (row >= static_cast<Index>(nrows_) ||
            col >= static_cast<Index>(outer_indices_.size() - 1)) {
            spdlog::warn(
                "Out of range issue detected! \"row\" and \"col\" should be less than \"nrows\" "
                "and \"ncols\" respectively: (row, col) = ({0:d}, {1:d}) while (nrows, ncols) = "
                "({2:d}, {3:d}).",
                row, col, nrows_, outer_indices_.size() - 1
            );
            return;
        }
        if (const auto search = std::find(
                inner_indices_.cbegin() + outer_indices_[col],
                inner_indices_.cbegin() + outer_indices_[col + 1], row
            );
            search != inner_indices_.cbegin() + outer_indices_[col + 1]) {
            const std::size_t inner_offset = search - inner_indices_.cbegin();
            if (value != 0.0) {
                values_[inner_offset] = value;
            } else {
                values_.erase(values_.begin() + inner_offset);
                inner_indices_.erase(inner_indices_.begin() + inner_offset);
                for (auto outer_it = outer_indices_.begin() + col + 1;
                     outer_it != outer_indices_.end(); ++outer_it) {
                    --(*outer_it);
                }
            }
            return;
        } else {
            if (value != 0.0) {
                const std::size_t inner_offset =
                    std::upper_bound(
                        inner_indices_.cbegin() + outer_indices_[col],
                        inner_indices_.cbegin() + outer_indices_[col + 1], row
                    ) -
                    inner_indices_.cbegin();
                values_.insert(values_.begin() + inner_offset, value);
                inner_indices_.insert(inner_indices_.begin() + inner_offset, row);
                for (auto outer_it = outer_indices_.begin() + col + 1;
                     outer_it != outer_indices_.end(); ++outer_it) {
                    ++(*outer_it);
                }
            }
        }
        return;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto values() const noexcept -> const std::vector<Scalar>& {
        return values_;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto innerIndices() const noexcept -> const std::vector<Index>& {
        return inner_indices_;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto outerIndices() const noexcept -> const std::vector<Index>& {
        return outer_indices_;
    }

    [[using gnu: always_inline]]
    auto clear() noexcept -> void {
        values_.clear();
        inner_indices_.clear();
        std::fill(outer_indices_.begin(), outer_indices_.end(), 0);
        return;
    }

  private:
    std::size_t nrows_{0};
    std::vector<Scalar> values_{};
    std::vector<Index> inner_indices_{};
    std::vector<Index> outer_indices_{};
};

} // namespace osqp
