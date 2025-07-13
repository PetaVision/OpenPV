set(SRC_CPP
  src/main.cpp
)

pv_add_test(PARAMS repTest norepTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
