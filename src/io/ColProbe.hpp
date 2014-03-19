/*
 * ColProbe.hpp
 *
 *  Created on: Nov 25, 2010
 *      Author: pschultz
 */

#ifndef COLPROBE_HPP_
#define COLPROBE_HPP_

#include <string.h>
#include "../columns/HyPerCol.hpp"

namespace PV {

class ColProbe {
public:
    ColProbe(const char * probeName, HyPerCol * hc);
    virtual ~ColProbe();

    int ioParams(enum ParamsIOFlag ioFlag);
    virtual int outputState(double time, HyPerCol * hc) {return PV_SUCCESS;}
    const char * getColProbeName() { return colProbeName; }

protected:
    HyPerCol * parentCol;
    PV_Stream * stream;
    char * colProbeName;

    ColProbe();
    int initialize(const char * probeName, HyPerCol * hc);
    int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);
    int initialize_stream(const char * filename);
    int setColProbeName(const char * name);

private:
    int initialize_base();
}; // end class ColProbe

}  // end namespace PV

#endif /* COLPROBE_HPP_ */
