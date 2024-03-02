add_header_only_library(qp_problem qp_problem.hpp)
target_link_libraries(qp_problem
  INTERFACE
    coo_matrix
    lil_matrix
)

add_header_only_library(osqp_solver osqp_solver.hpp)
target_link_libraries(osqp_solver
  INTERFACE
    osqpstatic
    csc_matrix
    qp_problem
)

install_libraries(
  qp_problem
  osqp_solver
)

add_subdirectory(sparse_matrix)
