set(DEMOS 
    # binocDemo
   HeatMapLocalization 
   openav
)

foreach(subdir ${DEMOS})
   message(STATUS "adding subdir ${subdir}")
   add_subdirectory(${subdir})
endforeach()
