/**
 * @file qp_problem.h
 * @author Houchen Li (houchen_li@hotmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-28
 *
 * @copyright Copyright (c) 2023 Houchen Li
 *            All rights reserved.
 *
 */

#pragma once

#include <concepts>
#include <iterator>
#include <numeric>
#include <unordered_map>

#include "sparse_matrix/coo_matrix.hpp"
#include "sparse_matrix/lil_matrix.hpp"

namespace osqp {

template <std::floating_point Scalar = double, std::integral Index = int>
class [[nodiscard]] QpProblem final {
    friend class OsqpSolver;

  public:
    [[using gnu: always_inline]] QpProblem(std::size_t num_vars, std::size_t num_cons) noexcept
        : num_vars_{static_cast<Index>(num_vars)}, num_cons_{static_cast<Index>(num_cons)},
          objective_matrix_{num_vars, num_vars}, objective_vector_(num_vars, 0.0),
          constrain_matrix_{num_cons, num_vars},
          lower_bounds_(num_cons, std::numeric_limits<Scalar>::lowest()),
          upper_bounds_(num_cons, std::numeric_limits<Scalar>::max()) {}

    QpProblem() noexcept = default;
    QpProblem(const QpProblem& other) noexcept = default;
    QpProblem(QpProblem&& other) noexcept = default;
    auto operator=(const QpProblem& other) noexcept -> QpProblem& = default;
    auto operator=(QpProblem&& other) noexcept -> QpProblem& = default;
    ~QpProblem() noexcept = default;

    [[using gnu: always_inline]]
    auto resize(std::size_t num_vars, std::size_t num_cons) noexcept -> void {
        num_vars_ = static_cast<Index>(num_vars);
        num_cons_ = static_cast<Index>(num_cons);
        objective_matrix_.resize(num_vars, num_vars);
        objective_vector_.resize(num_vars, 0.0);
        constrain_matrix_.resize(num_cons, num_vars);
        lower_bounds_.resize(num_cons, std::numeric_limits<Scalar>::lowest());
        upper_bounds_.resize(num_cons, std::numeric_limits<Scalar>::max());
        return;
    }

    [[using gnu: always_inline]]
    auto clear() noexcept -> void {
        num_vars_ = 0;
        num_cons_ = 0;
        objective_matrix_.clear();
        std::fill(objective_vector_.begin(), objective_vector_.end(), 0.0);
        constrain_matrix_.clear();
        std::fill(
            lower_bounds_.begin(), lower_bounds_.end(), std::numeric_limits<Scalar>::lowest()
        );
        std::fill(upper_bounds_.begin(), upper_bounds_.end(), std::numeric_limits<Scalar>::max());
        return;
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto num_variables() const noexcept -> std::size_t {
        return objective_matrix_.nrows();
    }

    [[using gnu: pure, always_inline]] [[nodiscard]]
    auto num_constraints() const noexcept -> std::size_t {
        return constrain_matrix_.nrows();
    }

    [[using gnu: pure]]
    auto cost(std::contiguous_iterator auto first, std::sentinel_for<decltype(first)> auto last)
        const noexcept -> Scalar
        requires std::floating_point<typename std::iterator_traits<decltype(first)>::value_type>
    {
        if (last - first != num_vars_) {
            spdlog::warn(
                "Mismatch size detected! The size of state_vec and num_variables has to be "
                "identical: state_vec.size() = {0:d} while num_variables() = {1:d}.",
                last - first, num_vars_
            );
            return 0.0;
        }
        Scalar cost{0.0};
        const std::vector<Scalar>& values{objective_matrix_.values()};
        const std::vector<Index>& row_indices{objective_matrix_.rowIndices()};
        const std::vector<Index>& col_indices{objective_matrix_.colIndices()};
        const Index values_size = values.size();
        for (Index i{0}; i < values_size; ++i) {
            const Index row = row_indices[i];
            const Index col = col_indices[i];
            if (row != col) {
                cost += values[i] * first[row] * first[col];
            } else {
                cost += values[i] * first[row] * first[col] * 0.5;
            }
        }
        for (Index i{0}; i < num_vars_; ++i) {
            cost += objective_vector_[i] * first[i];
        }
        return cost;
    }

    [[using gnu: pure]]
    auto validate(std::contiguous_iterator auto first, std::sentinel_for<decltype(first)> auto last)
        const noexcept -> bool
        requires std::floating_point<typename std::iterator_traits<decltype(first)>::value_type>
    {
        if (last - first != num_vars_) {
            spdlog::warn(
                "Mismatch size detected! The size of state_vec and num_variables has to be "
                "identical: state_vec.size() = {0:d} while num_variables() = {1:d}.",
                last - first, num_vars_
            );
            return false;
        }
        for (Index i{0}; i < num_cons_; ++i) {
            Scalar inner_prod{0.0};
            const std::vector<Scalar>& row_values_ = constrain_matrix_.rowValues(i);
            const std::vector<Index>& row_indices = constrain_matrix_.rowIndices(i);
            for (Index j{0}; j < num_vars_; ++j) {
                inner_prod += row_values_[j] * first[row_indices[j]];
            }
            if (inner_prod < lower_bounds_[i] || inner_prod > upper_bounds_[i]) {
                return false;
            }
        }
        return true;
    }

    [[using gnu: always_inline, hot]]
    auto addQuadCostTerm(Index row, Index col, Scalar coeff) noexcept -> void {
        if (row >= num_vars_ || col >= num_vars_) {
            spdlog::warn(
                "Out-of-range error detected! The index ({0:d},{1:d}) is not in the allowed range: "
                "the allowed range is ({2:d},{3:d}).",
                row, col, objective_matrix_.nrows(), objective_matrix_.ncols()
            );
            return;
        }
        if (coeff == 0.0) {
            return;
        }
        if (row < col) {
            coeff += objective_matrix_.coeff(row, col);
            objective_matrix_.updateCoeff(row, col, coeff);
        } else if (row == col) {
            coeff *= 2.0;
            coeff += objective_matrix_.coeff(row, col);
            objective_matrix_.updateCoeff(row, col, coeff);
        } else {
            spdlog::warn(
                "The objective matrix has to be a upper triangular sparse matrix which requires "
                "row_index < col_index. row_index: {0:d}, col_index: {1:d}. This term is updated "
                "to ({1:d}, {0:d}) instead.",
                row, col
            );
            coeff += objective_matrix_.coeff(col, row);
            objective_matrix_.updateCoeff(col, row, coeff);
        }
        return;
    }

    [[using gnu: always_inline, hot]]
    auto updateQuadCostTerm(Index row, Index col, Scalar coeff) noexcept -> void {
        if (row >= num_vars_ || col >= num_vars_) {
            spdlog::warn(
                "Out-of-range error detected! The index ({0:d},{1:d}) is not in the allowed range: "
                "the allowed range is ({2:d},{3:d}).",
                row, col, objective_matrix_.nrows(), objective_matrix_.ncols()
            );
            return;
        }
        if (row < col) {
            objective_matrix_.updateCoeff(row, col, coeff);
        } else if (row == col) {
            objective_matrix_.updateCoeff(row, col, coeff * 2.0);
        } else {
            spdlog::warn(
                "The objective matrix has to be a upper triangular sparse matrix which requires "
                "row_index < col_index. row_index: {0:d}, col_index: {1:d}. This term is updated "
                "to ({1:d}, {0:d}) instead.",
                row, col
            );
            objective_matrix_.updateCoeff(col, row, coeff);
        }
        return;
    }

    [[using gnu: always_inline, hot]]
    auto addLinCostTerm(Index row, Scalar coeff) noexcept -> void {
        if (row >= num_vars_) {
            spdlog::warn(
                "Out-of-range error detected! The input row has to be in the allowed range: row = "
                "{0:d} while max_row = {1:d}).",
                row, objective_vector_.size()
            );
            return;
        }
        objective_vector_[row] += coeff;
        return;
    }

    [[using gnu: always_inline, hot]]
    auto updateLinCostTerm(Index row, Scalar coeff) noexcept -> void {
        if (row >= num_vars_) {
            spdlog::warn(
                "Out-of-range error detected! The input row has to be in the allowed range: row = "
                "{0:d} while max_row = {1:d}).",
                row, objective_vector_.size()
            );
            return;
        }
        objective_vector_[row] = coeff;
        return;
    }

    [[using gnu: always_inline, hot]]
    auto addClampCostTerm(
        std::unordered_map<Index, Scalar> constrain_vec, Scalar offset, Scalar linear_coeff,
        Scalar quadratic_coeff = 0.0
    ) noexcept -> void {
        objective_matrix_.resize(num_vars_ + 1, num_vars_ + 1);
        objective_matrix_.updateCoeff(num_vars_, num_vars_, quadratic_coeff * 2.0);
        objective_vector_.push_back(linear_coeff);
        constrain_matrix_.resize(num_cons_ + 2, num_vars_ + 1);
        constrain_matrix_.updateRow(num_cons_, {{num_vars_, 1.0}});
        lower_bounds_.push_back(0.0);
        upper_bounds_.push_back(std::numeric_limits<Scalar>::max());
        constrain_vec.emplace(num_vars_, -1.0);
        constrain_matrix_.updateRow(num_cons_ + 1, constrain_vec);
        lower_bounds_.push_back(std::numeric_limits<Scalar>::lowest());
        upper_bounds_.push_back(offset);
        num_vars_ += 1;
        num_cons_ += 2;
        return;
    }

    [[using gnu: always_inline, hot]]
    auto addConstrainTerm(
        const std::unordered_map<Index, Scalar>& constrain_vec, Scalar lower_bound,
        Scalar upper_bound
    ) noexcept -> void {
        constrain_matrix_.resize(num_cons_ + 1, constrain_matrix_.ncols());
        constrain_matrix_.updateRow(num_cons_, constrain_vec);
        lower_bounds_.push_back(lower_bound);
        upper_bounds_.push_back(upper_bound);
        num_cons_ += 1;
        return;
    }

    [[using gnu: always_inline, hot]]
    auto updateConstrainTerm(
        Index row, const std::unordered_map<Index, Scalar>& constrain_vec, Scalar lower_bound,
        Scalar upper_bound
    ) noexcept -> void {
        if (row >= num_cons_) {
            spdlog::warn(
                "Out-of-range error detected! The input row has to be in the allowed range: row = "
                "{0:d} while max_row = {1:d}).",
                row, constrain_matrix_.nrows()
            );
            return;
        }
        constrain_matrix_.updateRow(row, constrain_vec);
        lower_bounds_[row] = lower_bound;
        upper_bounds_[row] = upper_bound;
        return;
    }

  private:
    Index num_vars_{0};
    Index num_cons_{0};
    CooMatrix<Scalar, Index> objective_matrix_{};
    std::vector<Scalar> objective_vector_{};
    LilMatrix<Scalar, Index> constrain_matrix_{};
    std::vector<Scalar> lower_bounds_{};
    std::vector<Scalar> upper_bounds_{};
};

} // namespace osqp
