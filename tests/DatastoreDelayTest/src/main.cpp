/*
 * DatastoreDelayTest.cpp
 *
 */

// using DatastoreDelayLayer, an input layer is filled with
// random data with the property that summing across four
// adjacent rows gives zeroes.
//
// On each timestep the data is rotated by one column.
// The input goes through four connections, with delays 0,1,2,3,
// each on the excitatory channel.
//
// The output layer should therefore be all zeros.

#include "DatastoreDelayTestLayer.hpp"
#include "DatastoreDelayTestProbe.hpp"
#include "columns/buildandrun.hpp"
#include <utils/PVLog.hpp>

int main(int argc, char *argv[]) {

   int status;
   PV_Init initObj(&argc, &argv, false /*allowUnrecognizedArguments*/);
   initObj.registerKeyword("DatastoreDelayTestLayer", Factory::create<DatastoreDelayTestLayer>);
   initObj.registerKeyword("DatastoreDelayTestProbe", Factory::create<DatastoreDelayTestProbe>);
   if (initObj.getParams() == nullptr) {
      initObj.setParams("input/DatastoreDelayTest.params");
   }
   status = buildandrun(&initObj);
   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
