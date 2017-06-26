/*
 * FeedbackConn.hpp
 *
 *  Created on: Oct 27, 2010
 *      Author: pschultz
 */

#ifndef FEEDBACKCONN_HPP_
#define FEEDBACKCONN_HPP_

#include <assert.h>
#include <string.h>

#include "TransposeConn.hpp"
#include "io/fileio.hpp"
#include "io/io.hpp"

namespace PV {

class FeedbackConn : public TransposeConn {
  public:
   FeedbackConn(const char *name, HyPerCol *hc);

  protected:
   FeedbackConn();
   int initialize_base();
   int initialize(const char *name, HyPerCol *hc);
   void ioParam_preLayerName(enum ParamsIOFlag ioFlag) override;
   void ioParam_postLayerName(enum ParamsIOFlag ioFlag) override;

   virtual int setPreAndPostLayerNames() override;
   virtual int handleMissingPreAndPostLayerNames() override;
}; // end class FeedbackConn

} // end of block for namespace PV

#endif /* FEEDBACKCONN_HPP_ */
