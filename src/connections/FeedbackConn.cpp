/* FeedbackConn.cpp
 *
 * Created on: Nov 15, 2010
 *     Author: peteschultz
 */

#include "FeedbackConn.hpp"
#include "columns/HyPerCol.hpp"
#include "components/FeedbackConnectionData.hpp"

namespace PV {

FeedbackConn::FeedbackConn(char const *name, PVParams *params, Communicator *comm) {
   initialize(name, params, comm);
}

FeedbackConn::FeedbackConn() {}

FeedbackConn::~FeedbackConn() {}

void FeedbackConn::initialize(char const *name, PVParams *params, Communicator *comm) {
   TransposeConn::initialize(name, params, comm);
}

ConnectionData *FeedbackConn::createConnectionData() {
   return new FeedbackConnectionData(name, parameters(), mCommunicator);
}

} // namespace PV
