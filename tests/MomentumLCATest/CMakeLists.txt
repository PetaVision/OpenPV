set(SRC_CPP
  src/main.cpp
)

pv_add_test(FLAGS "-c checkpoints/Checkpoint06 --testall" SRCFILES ${SRC_CPP} ${SRC_HPP} ${SRC_C} ${SRC_H})
