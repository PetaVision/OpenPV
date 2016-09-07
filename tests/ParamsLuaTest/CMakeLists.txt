set(SRC_CPP
  ${TESTS_SHARED_DIR}/ColumnArchive.cpp
  src/main.cpp
)

set(SRC_HPP
  ${TESTS_SHARED_DIR}/ColumnArchive.hpp
)

if (PV_USE_LUA)
  pv_add_test(NO_PARAMS ParamsLuaTest SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
endif (PV_USE_LUA)
