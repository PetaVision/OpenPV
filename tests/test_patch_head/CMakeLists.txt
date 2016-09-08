set(SRC_CPP
  src/test_patch_head.cpp
)

pv_add_test(NO_PARAMS NO_MPI SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
target_link_libraries(${BASE_NAME} m)
