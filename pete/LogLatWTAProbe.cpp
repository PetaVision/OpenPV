/*
 * LogLatWTAProbe.cpp
 *
 * A derived class of LayerFunctionProbe that uses LogLatWTAFunction
 *
 *  Created on: Apr 26, 2011
 *      Author: peteschultz
 */

#include "LogLatWTAProbe.hpp"

namespace PV {

LogLatWTAProbe::LogLatWTAProbe(const char * msg) : LayerFunctionProbe(msg) {
    function = new LogLatWTAFunction(msg);
}
LogLatWTAProbe::LogLatWTAProbe(const char * filename, HyPerCol * hc, const char * msg) : LayerFunctionProbe(filename, hc, msg) {
    function = new LogLatWTAFunction(msg);
}

LogLatWTAProbe::~LogLatWTAProbe() {
    delete function;
}

int LogLatWTAProbe::outputState(float time, HyPerLayer * l) {
    int nk = l->getNumNeurons();
    pvdata_t sum = function->evaluate(time, l);

    fprintf(fp, "%st = %6.3f numNeurons = %8d Lateral Competition Penalty = %f\n", msg, time, nk, sum);
    fflush(fp);

    return EXIT_SUCCESS;
}

}  // end of namespace PV

