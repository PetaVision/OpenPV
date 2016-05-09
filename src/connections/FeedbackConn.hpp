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
    FeedbackConn(const char * name, HyPerCol * hc);

protected:
    FeedbackConn();
    int initialize_base();
    int initialize(const char * name, HyPerCol * hc);
    void ioParam_preLayerName(enum ParamsIOFlag ioFlag);
    void ioParam_postLayerName(enum ParamsIOFlag ioFlag);

    virtual int setPreAndPostLayerNames();
    virtual int handleMissingPreAndPostLayerNames();
}; // end class FeedbackConn

BaseObject * createFeedbackConn(char const * name, HyPerCol * hc);

}  // end of block for namespace PV

#endif /* FEEDBACKCONN_HPP_ */
