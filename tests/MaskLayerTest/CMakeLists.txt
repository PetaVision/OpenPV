set(SRC_CPP
  src/MaskTest.cpp
  src/MaskTestLayer.cpp
)

set(SRC_HPP
  src/MaskTestLayer.hpp
)

pv_add_test(PARAMS maskTest invertMaskTest maskTestFeatures maskTestNoFeatures SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
