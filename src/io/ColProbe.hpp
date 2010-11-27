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
    ColProbe();
    ColProbe(const char * filename);
    virtual ~ColProbe();

    virtual int outputState(float time, HyPerCol * hc) {return EXIT_SUCCESS;}

protected:
    FILE * fp;
}; // end class ColProbe

}  // end namespace PV

#endif /* COLPROBE_HPP_ */
