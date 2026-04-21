set(SRC_CPP
  src/main.cpp
)

set(PARAMS_LIST ByFile Random SyncLayer)

pv_add_test(PARAMS ${PARAMS_LIST} SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
