/**
 * @file osqp_solver_test.cpp
 * @author Houchen Li (houchen_li@hotmail.com)
 * @brief
 * @version 0.1
 * @date 2023-07-21
 *
 * @copyright Copyright (c) 2023 Houchen Li
 *            All rights reserved.
 *
 */

#include "osqp_solver.hpp"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

namespace osqp {

constexpr double kEpsilon{1E-8};

/**
 * @brief Here we construct a simple Quadratic programming model to test.
 *        cost:
 *            f(x0, x1) = 2 * x0^2 + x1^2 + x0*x1 + x0 + x1
 *        constraints:
 *            1 <= x0 + x1 <= 1
 *            0 <= x0 <= 0.7
 *            0 <= x1 <= 0.7
 *
 */
class OsqpSolverTestFixture {
  public:
    OsqpSolverTestFixture() : qp_problem(2, 3), solver() {
        solver.settings.polishing = 1;

        qp_problem.updateQuadCostTerm(0, 0, 2.0);
        qp_problem.updateQuadCostTerm(1, 1, 1.0);
        qp_problem.updateQuadCostTerm(0, 1, 1.0);
        qp_problem.updateLinCostTerm(0, 1.0);
        qp_problem.updateLinCostTerm(1, 1.0);
        qp_problem.updateConstrainTerm(0, {{0, 1.0}, {1, 1.0}}, 1.0, 1.0);
        qp_problem.updateConstrainTerm(1, {{0, 1.0}}, 0.0, 0.7);
        qp_problem.updateConstrainTerm(2, {{1, 1.0}}, 0.0, 0.7);

        // Run cold start
        result = solver.solve(qp_problem);

        CHECK_EQ(result.prim_vars[0], doctest::Approx(0.3).epsilon(1e-7));
        CHECK_EQ(result.prim_vars[1], doctest::Approx(0.7).epsilon(1e-7));
        CHECK_EQ(result.dual_vars[0], doctest::Approx(-2.9).epsilon(1e-7));
        CHECK_EQ(result.dual_vars[1], 0.0);
        CHECK_EQ(result.dual_vars[2], doctest::Approx(0.2).epsilon(1e-6));
        CHECK_EQ(result.info.iter, 25);
    }

  protected:
    QpProblem<double, int> qp_problem{};
    OsqpResult result{};
    OsqpSolver solver{};
};

TEST_CASE_FIXTURE(OsqpSolverTestFixture, "WarmStart") {
    const std::vector<double> prim_vars_0{0.3, 0.7};
    const std::vector<double> dual_vars_0{-2.9, 0.0, 0.2};

    result = solver.solve(qp_problem, prim_vars_0, dual_vars_0);

    CHECK_EQ(result.prim_vars[0], doctest::Approx(0.3).epsilon(1e-7));
    CHECK_EQ(result.prim_vars[1], doctest::Approx(0.7).epsilon(1e-7));
    CHECK_EQ(result.dual_vars[0], doctest::Approx(-2.9).epsilon(1e-7));
    CHECK_EQ(result.dual_vars[1], 0.0);
    CHECK_EQ(result.dual_vars[2], doctest::Approx(0.2).epsilon(1e-6));
    CHECK_EQ(result.info.iter, 25);
}

} // namespace osqp
