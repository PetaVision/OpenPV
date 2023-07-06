set(SRC_CPP
  src/main.cpp
)

pv_add_test(
   PARAMS SharedToLocalPatchWeightsTest ExtraMarginWidth
   SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
