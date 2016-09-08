set(SRC_CPP
  src/ImagePvpTestLayer.cpp
  src/ImageSystemTest.cpp
  src/ImageTestLayer.cpp
  src/MoviePvpTestLayer.cpp
  src/MovieTestLayer.cpp
)

set(SRC_HPP
  src/ImagePvpTestLayer.hpp
  src/ImageTestLayer.hpp
  src/MoviePvpTestLayer.hpp
  src/MovieTestLayer.hpp
)

pv_add_test(PARAMS ImageFileIO ImagePvpFileIO ImagePvpFileIOSparse MovieFileIO MoviePvpFileIO SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
pv_add_test(PARAMS batchMovieFileIO MIN_MPI_COPIES 2 MPI_ONLY FLAGS "-batchwidth 2" BASE_NAME ImageSystemTest_batchMovieFileIO SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
