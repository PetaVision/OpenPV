set(SRC_CPP
  src/NormalizeL3.cpp
  src/NormalizeSubclassSystemTest.cpp
)

set(SRC_HPP
  src/NormalizeL3.hpp
)

pv_add_test(NO_MPI SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
