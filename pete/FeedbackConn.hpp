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

#include "GenerativeConn.hpp"
#include "../PetaVision/src/io/io.h"
#include "../PetaVision/src/io/fileio.hpp"

namespace PV {

class FeedbackConn : public GenerativeConn {
public:
    FeedbackConn(const char * name, HyPerCol *hc, int channel,
        GenerativeConn * ffconn);

    int initialize(const char * name, HyPerCol * hc, int channel,
        const char * filename, GenerativeConn * ffconn);
    inline GenerativeConn * getFeedforwardConn() {return feedforwardConn;}
    int updateWeights(int axonID);

protected:
    int setPatchSize(const char * filename);
    PVPatch ** initializeWeights(PVPatch ** patches, int numPatches,
          const char * filename);
    int transposeKernels();
    GenerativeConn * feedforwardConn;
};

}  // end of block for namespace PV

#endif /* FEEDBACKCONN_HPP_ */
