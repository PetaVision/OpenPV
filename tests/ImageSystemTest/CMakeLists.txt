set(SRC_CPP
  src/ImagePvpTestBuffer.cpp
  src/ImagePvpTestLayer.cpp
  src/ImageSystemTest.cpp
  src/ImageTestLayer.cpp
  src/MoviePvpTestBuffer.cpp
  src/MoviePvpTestLayer.cpp
  src/MovieTestBuffer.cpp
  src/MovieTestLayer.cpp
)

set(SRC_HPP
  src/ImagePvpTestBuffer.hpp
  src/ImagePvpTestLayer.hpp
  src/ImageTestLayer.hpp
  src/MoviePvpTestBuffer.hpp
  src/MoviePvpTestLayer.hpp
  src/MovieTestBuffer.hpp
  src/MovieTestLayer.hpp
)

pv_add_test(PARAMS ImageFileIO ImagePvpFileIO ImagePvpFileIOSparse MovieFileIO MoviePvpFileIO SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
pv_add_test(PARAMS batchMovieFileIO MIN_MPI_COPIES 2 MPI_ONLY FLAGS "-batchwidth 2" BASE_NAME ImageSystemTest_batchMovieFileIO SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
