set(SRC_CPP
  src/main.cpp
  src/FixedImageSequenceByFile.cpp
  src/FixedImageSequenceByList.cpp
  src/FixedImageSequence.cpp
)

set(SRC_HPP
  src/FixedImageSequenceByFile.hpp
  src/FixedImageSequenceByList.hpp
  src/FixedImageSequence.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
