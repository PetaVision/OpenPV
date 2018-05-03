/* FeedbackConn.cpp
 *
 * Created on: Nov 15, 2010
 *     Author: peteschultz
 */

#include "FeedbackConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/FeedbackConnectionData.hpp"

namespace PV {

FeedbackConn::FeedbackConn(char const *name, HyPerCol *hc) { initialize(name, hc); }

FeedbackConn::FeedbackConn() {}

FeedbackConn::~FeedbackConn() {}

int FeedbackConn::initialize(char const *name, HyPerCol *hc) {
   int status = TransposeConn::initialize(name, hc);
   return status;
}

ConnectionData *FeedbackConn::createConnectionData() {
   return new FeedbackConnectionData(name, parent);
}

} // namespace PV
