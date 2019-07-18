#include "connections/AffineCopyConn.hpp"

#include <columns/PV_Init.hpp>
#include <columns/buildandrun.hpp>

int main(int argc, char *argv[]) {
   AffineCopyConn::rotation_test(argv[1],argv[2],argv[3]);
   return 0;
}


