/*
 * FeedbackConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef FEEDBACKCONN_HPP_
#define FEEDBACKCONN_HPP_

#include <string.h>
#include <assert.h>

#include "TransposeConn.hpp"
#include "../io/io.h"
#include "../io/fileio.hpp"

namespace PV {

class FeedbackConn : public TransposeConn {
public:
    FeedbackConn();
    FeedbackConn(const char * name, HyPerCol * hc, const char * feedforwardConnName);

    int initialize_base();
    int initialize(const char * name, HyPerCol * hc, const char * feedforwardConnName);

protected:
    PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
          const char * filename);
    virtual int handleMissingPreAndPostLayerNames();
};

}  // end of block for namespace PV

#endif /* FEEDBACKCONN_HPP_ */
