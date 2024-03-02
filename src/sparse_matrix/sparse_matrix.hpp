/**
 * @file sparse_matrix.hpp
 * @author Houchen Li (houchen_li@hotmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-19
 *
 * @copyright Copyright (c) 2023 Houchen Li
 *            All rights reserved.
 *
 */

#pragma once

#include <concepts>
#include <unordered_map>
#include <vector>

namespace osqp {

template <typename DerivedMatrix, std::floating_point Scalar = double, std::integral Index = int>
class [[nodiscard]] SparseMatrix {
  public:
    SparseMatrix() noexcept = default;
    SparseMatrix(const SparseMatrix& other) noexcept = default;
    SparseMatrix(SparseMatrix&& other) noexcept = default;
    auto operator=(const SparseMatrix& other) noexcept -> SparseMatrix& = default;
    auto operator=(SparseMatrix&& other) noexcept -> SparseMatrix& = default;
    virtual ~SparseMatrix() noexcept = 0;

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto nrows() const noexcept -> std::size_t {
        return underlying()->nrows();
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto ncols() const noexcept -> std::size_t {
        return underlying()->ncols();
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto nnzs() const noexcept -> std::size_t {
        return underlying()->nnzs();
    }

    [[using gnu: always_inline]]
    auto resize(std::size_t nrows, std::size_t ncols) noexcept -> void {
        underlying()->resize(nrows, ncols);
        return;
    }

    [[using gnu: pure, always_inline]]
    auto coeff(Index row, Index col) const noexcept -> Scalar {
        return underlying()->coeff(row, col);
    }

    [[using gnu: pure, always_inline]]
    auto
    operator()(Index row, Index col) const noexcept -> Scalar {
        return coeff(row, col);
    }

  private:
    [[using gnu: pure, always_inline]]
    auto underlying() noexcept -> DerivedMatrix* {
        return static_cast<DerivedMatrix*>(this);
    }

    [[using gnu: pure, always_inline]]
    auto underlying() const noexcept -> const DerivedMatrix* {
        return static_cast<const DerivedMatrix*>(this);
    }
};

template <typename DerivedMatrix, std::floating_point Scalar, std::integral Index>
inline SparseMatrix<DerivedMatrix, Scalar, Index>::~SparseMatrix() noexcept = default;

} // namespace osqp
