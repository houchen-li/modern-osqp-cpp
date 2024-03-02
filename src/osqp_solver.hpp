/**
 * @file osqp_solver.h
 * @author Houchen Li (houchen_li@hotmail.com)
 * @brief
 * @version 0.1
 * @date 2023-10-29
 *
 * @copyright Copyright (c) 2023 Houchen Li
 *            All rights reserved.
 *
 */

#pragma once

extern "C" {
#include "osqp.h"
}

#include <concepts>
#include <format>
#include <functional>
#include <stdexcept>
#include <utility>

#include "qp_problem.hpp"
#include "sparse_matrix/csc_matrix.hpp"

namespace osqp {

namespace detail {

class [[nodiscard]] ExecOnExit final {
  public:
    using Callback = std::function<void()>;
    ExecOnExit(Callback callback) noexcept : callback_(std::move(callback)) {}

    ExecOnExit() noexcept = delete;
    ExecOnExit(const ExecOnExit& other) noexcept = delete;
    ExecOnExit(ExecOnExit&& other) noexcept = delete;
    auto operator=(const ExecOnExit& other) noexcept -> ExecOnExit& = delete;
    auto operator=(ExecOnExit&& other) noexcept -> ExecOnExit& = delete;
    ~ExecOnExit() { callback_(); }

  private:
    Callback callback_;
};

} // namespace detail

using OsqpSettings = OSQPSettings;
using OsqpInfo = OSQPInfo;

struct [[nodiscard]] OsqpResult final {
    std::vector<OSQPFloat> prim_vars;
    std::vector<OSQPFloat> prim_inf_cert;
    std::vector<OSQPFloat> dual_vars;
    std::vector<OSQPFloat> dual_inf_cert;
    OsqpInfo info;
};

class [[nodiscard]] OsqpSolver final {
  public:
    [[using gnu: always_inline]] OsqpSolver() noexcept { osqp_set_default_settings(&settings); }
    [[using gnu: always_inline]] explicit OsqpSolver(const OsqpSettings& c_settings) noexcept
        : settings{c_settings} {}

    OsqpSolver(const OsqpSolver& other) noexcept = delete;
    OsqpSolver(OsqpSolver&& other) noexcept = delete;
    auto operator=(const OsqpSolver& other) noexcept -> OsqpSolver& = delete;
    auto operator=(OsqpSolver&& other) noexcept -> OsqpSolver& = delete;
    ~OsqpSolver() noexcept = default;

    template <std::floating_point Scalar, std::integral Index>
    [[using gnu: pure, flatten, leaf]] [[nodiscard]]
    auto solve(
        const QpProblem<Scalar, Index>& qp_problem, const std::vector<Scalar>& prim_vars_0 = {},
        const std::vector<Scalar>& dual_vars_0 = {}
    ) const -> OsqpResult {
#ifndef NDEBUG
        if (!prim_vars_0.empty() && prim_vars_0.size() != qp_problem.num_variables()) {
            std::string error_msg = std::format(
                "Osqp runtime error detected! Size of prim_vars_0_ and num_vars must be identical: "
                "prim_vars_0_.size() = {0:d} while num_vars = {1:d}",
                prim_vars_0.size(), qp_problem.num_variables()
            );
            throw std::invalid_argument(std::move(error_msg));
        }
        if (!dual_vars_0.empty() && dual_vars_0.size() != qp_problem.num_constraints()) {
            std::string error_msg = std::format(
                "Osqp runtime error detected! Size of dual_vars_0_ and num_cons must be identical: "
                "dual_vars_0_.size() = {0:d} while num_cons = {1:d}",
                dual_vars_0.size(), qp_problem.num_constraints()
            );
            throw std::invalid_argument(std::move(error_msg));
        }
#endif

        OSQPInt exit_flag{0};
        OSQPSolver* solver = static_cast<OSQPSolver*>(malloc(sizeof(OSQPSolver)));
        OSQPCscMatrix* P = static_cast<OSQPCscMatrix*>(malloc(sizeof(OSQPCscMatrix)));
        OSQPCscMatrix* A = static_cast<OSQPCscMatrix*>(malloc(sizeof(OSQPCscMatrix)));

        detail::ExecOnExit exec_on_exit{[&solver, &P, &A]() -> void {
            osqp_cleanup(solver);
            free(P);
            free(A);
            return;
        }};

        const OSQPInt num_vars = static_cast<OSQPInt>(qp_problem.num_variables());
        const OSQPInt num_cons = static_cast<OSQPInt>(qp_problem.num_constraints());
        CscMatrix<OSQPFloat, OSQPInt> objective_matrix{qp_problem.objective_matrix_};
        std::vector<OSQPFloat> objective_vector{qp_problem.objective_vector_};
        CscMatrix<OSQPFloat, OSQPInt> constrain_matrix{qp_problem.constrain_matrix_};
        std::vector<OSQPFloat> lower_bounds{qp_problem.lower_bounds_};
        std::vector<OSQPFloat> upper_bounds{qp_problem.upper_bounds_};

        csc_set_data(
            P, num_vars, num_vars, static_cast<OSQPInt>(objective_matrix.nnzs()),
            const_cast<OSQPFloat*>(objective_matrix.values().data()),
            const_cast<OSQPInt*>(objective_matrix.innerIndices().data()),
            const_cast<OSQPInt*>(objective_matrix.outerIndices().data())
        );
        csc_set_data(
            A, num_cons, num_vars, static_cast<OSQPInt>(constrain_matrix.nnzs()),
            const_cast<OSQPFloat*>(constrain_matrix.values().data()),
            const_cast<OSQPInt*>(constrain_matrix.innerIndices().data()),
            const_cast<OSQPInt*>(constrain_matrix.outerIndices().data())
        );

        exit_flag = osqp_setup(
            &solver, P, objective_vector.data(), A, lower_bounds.data(), upper_bounds.data(),
            num_cons, num_vars, &settings
        );
        if (exit_flag) {
            std::string error_msg = std::format(
                "Osqp runtime error detected! Not able to setup a OSQPSolver: exit_flag = {0:d}.",
                exit_flag
            );
            throw std::runtime_error(std::move(error_msg));
        }

        // Set warm start
        if (!prim_vars_0.empty() && !dual_vars_0.empty()) {
            exit_flag = osqp_warm_start(solver, prim_vars_0.data(), dual_vars_0.data());
            if (exit_flag) {
                std::string error_msg = std::format(
                    "Osqp runtime error detected! Not able to set warm start for a OSQPSolver: "
                    "exit_flag = {0:d}.",
                    exit_flag
                );
                throw std::runtime_error(std::move(error_msg));
            }
        }

        exit_flag = osqp_solve(solver);
        if (exit_flag) {
            std::string error_msg = std::format(
                "Osqp runtime error detected! Not able to execute a OSQPSolver: exit_flag = {0:d}.",
                exit_flag
            );
            throw std::runtime_error(std::move(error_msg));
        }

        return OsqpResult{
            .prim_vars{solver->solution->x, solver->solution->x + num_vars},
            .prim_inf_cert{
                solver->solution->prim_inf_cert, solver->solution->prim_inf_cert + num_vars},
            .dual_vars{solver->solution->y, solver->solution->y + num_cons},
            .dual_inf_cert{
                solver->solution->dual_inf_cert, solver->solution->dual_inf_cert + num_cons},
            .info{*(solver->info)}};
    }

    OsqpSettings settings{};
};

} // namespace osqp
