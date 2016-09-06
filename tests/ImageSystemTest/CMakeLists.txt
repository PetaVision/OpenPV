pv_add_test(PARAMS ImageFileIO ImagePvpFileIO ImagePvpFileIOSparse MovieFileIO MoviePvpFileIO)
pv_add_test(PARAMS batchMovieFileIO MIN_MPI_COPIES 2 MPI_ONLY FLAGS "-batchwidth 2" BASE_NAME ImageSystemTest_batchMovieFileIO)
