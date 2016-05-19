set(DEMOS 
    # binocDemo
   HeatMapLocalization 
)

foreach(subdir ${DEMOS})
   message(STATUS "adding subdir ${subdir}")
   add_subdirectory(${subdir})
endforeach()
