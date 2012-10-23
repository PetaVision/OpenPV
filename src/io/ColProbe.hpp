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
    ColProbe(const char * filename, HyPerCol * hc);
    ColProbe(const char * probeName, const char * filename, HyPerCol * hc);
    virtual ~ColProbe();

    virtual int outputState(double time, HyPerCol * hc) {return PV_SUCCESS;}
    const char * getColProbeName() { return colProbeName; }

protected:
    FILE * fp;
    char * colProbeName;

    ColProbe();
    int initialize(const char * probeName, const char * filename, HyPerCol * hc);
    int initialize_path(const char * filename, HyPerCol * hc);
    int setColProbeName(const char * name);
}; // end class ColProbe

}  // end namespace PV

#endif /* COLPROBE_HPP_ */
