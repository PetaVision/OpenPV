if (PV_USE_GDAL)
   pv_add_test(PARAMS ImageOffsetTest ImagePvpOffsetTest)
else (PV_USE_GDAL)
   pv_add_test(PARAMS ImagePvpOffsetTest)
endif (PV_USE_GDAL)

