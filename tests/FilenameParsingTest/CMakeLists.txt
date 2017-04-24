set(SRC_CPP
  src/FilenameParsingProbe.cpp
  src/FilenameParsingTest.cpp
)

set(SRC_HPP
  src/FilenameParsingProbe.hpp
)

pv_add_test(SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
