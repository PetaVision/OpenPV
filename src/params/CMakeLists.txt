# Nicer for in-source builds. Puts the library into a subdir
set(LIBRARY_OUTPUT_PATH ${PV_LIBRARY_DIR})

set(PV_PARAMS_BUILD_DIR "${PV_LIBRARY_DIR}/params")
set(PV_PARSER_DIR "${PV_PARAMS_BUILD_DIR}/parser-generated")
set(PV_PARAM_LEXER_C "${PV_PARSER_DIR}/param_lexer.c")
set(PV_PARAM_PARSER_CPP "${PV_PARSER_DIR}/param_parser.cpp")
set(PV_PARAM_PARSER_HPP "${PV_PARSER_DIR}/param_parser.hpp")

set(PV_LIBRARY_NAME "pv")

# Create directory structure for lexer and parser output
file(MAKE_DIRECTORY ${PV_PARAMS_BUILD_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_PARAMS_BUILD_DIR})
file(MAKE_DIRECTORY ${PV_PARSER_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${PV_PARSER_DIR})

# Lex/Yacc for parser
if (NOT BISON_PARSER_DEFINED)
  find_package(BISON REQUIRED)
  BISON_TARGET(
    PARSER ${PV_SOURCE_DIR}/params/parser/params.y ${PV_PARAM_PARSER_CPP}
    DEFINES_FILE ${PV_PARAM_PARSER_HPP} COMPILE_FLAGS "-y")
endif (NOT BISON_PARSER_DEFINED)

if (NOT FLEX_SCANNER_DEFINED)
  find_package(FLEX REQUIRED)
  FLEX_TARGET(SCANNER ${PV_SOURCE_DIR}/params/parser/params.l ${PV_PARAM_LEXER_C})
  ADD_FLEX_BISON_DEPENDENCY(SCANNER PARSER)
endif (NOT FLEX_SCANNER_DEFINED)

set (PVLibSrcCpp ${PVLibSrcCpp}
   ${SUBDIR}/Parameter.cpp
   ${SUBDIR}/ParameterSweep.cpp
   ${SUBDIR}/ParamGroup.cpp
   ${SUBDIR}/ParamGroupList.cpp
   ${SUBDIR}/ParamsIO.cpp
   ${SUBDIR}/PVParams.cpp
)

set (PVLibSrcHpp ${PVLibSrcHpp}
   ${SUBDIR}/Parameter.hpp
   ${SUBDIR}/ParameterSweep.hpp
   ${SUBDIR}/ParamGroup.hpp
   ${SUBDIR}/ParamGroupList.hpp
   ${SUBDIR}/ParamsIO.hpp
   ${SUBDIR}/PVParams.hpp
)
