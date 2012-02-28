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
    FeedbackConn(const char * name, HyPerCol *hc, ChannelType channel,
        KernelConn * ffconn);

    int initialize_base();
    int initialize(const char * name, HyPerCol * hc, ChannelType channel, KernelConn * ffconn);

protected:
    int setPatchSize(const char * filename);
    PVPatch *** initializeWeights(PVPatch *** arbors, pvdata_t ** dataStart, int numPatches,
          const char * filename);

    KernelConn * feedforwardConn; // same as TransposeConn's originalConn; kept for backward compatibility
};

}  // end of block for namespace PV

#endif /* FEEDBACKCONN_HPP_ */
