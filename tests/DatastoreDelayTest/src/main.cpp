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

#include <columns/buildandrun.hpp>
#include <io/io.h>
#include "DatastoreDelayTestLayer.hpp"
#include "DatastoreDelayTestProbe.hpp"
#include <assert.h>

int main(int argc, char * argv[]) {

    int status;
    PV_Init initObj(&argc, &argv, false/*allowUnrecognizedArguments*/);
    initObj.registerKeyword("DatastoreDelayTestLayer", createDatastoreDelayTestLayer);
    initObj.registerKeyword("DatastoreDelayTestProbe", createDatastoreDelayTestProbe);
    PV_Arguments * arguments = initObj.getArguments();
    if (arguments->getParamsFile()==NULL) {
        arguments->setParamsFile("input/DatastoreDelayTest.params");
    }
    status = buildandrun(&initObj);
    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}
