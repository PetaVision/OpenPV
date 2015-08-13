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
    char const * getColProbeName() { return colProbeName; }
    char const * keyword();

protected:
    HyPerCol * parentCol;
    PV_Stream * stream;
    char * colProbeName;

    ColProbe();
    int initialize(const char * probeName, HyPerCol * hc);
    int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
    virtual void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);
    int initialize_stream(const char * filename);
    virtual int outputHeader() { return PV_SUCCESS; }
    int setColProbeName(const char * name);

private:
    int initialize_base();
}; // end class ColProbe

}  // end namespace PV

#endif /* COLPROBE_HPP_ */
