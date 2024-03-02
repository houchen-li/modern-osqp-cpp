add_header_only_library(sparse_matrix sparse_matrix.hpp)

add_header_only_library(coo_matrix coo_matrix.hpp)
target_link_libraries(coo_matrix
  INTERFACE
    sparse_matrix
)

add_header_only_library(lil_matrix lil_matrix.hpp)
target_link_libraries(lil_matrix
  INTERFACE
    sparse_matrix
)

add_header_only_library(csc_matrix csc_matrix.hpp)
target_link_libraries(csc_matrix
  INTERFACE
    sparse_matrix
)

add_header_only_library(csr_matrix csr_matrix.hpp)
target_link_libraries(csr_matrix
  INTERFACE
    sparse_matrix
)

install_libraries(
  coo_matrix
  lil_matrix
  csc_matrix
  csr_matrix
)
