/* FeedbackConn.cpp
 *
 * Created on: Oct 27, 2010
 *     Author: peteschultz
 */

#ifndef FEEDBACKCONN_HPP_
#define FEEDBACKCONN_HPP_

#include "components/OriginalConnNameParam.hpp"
#include "connections/TransposeConn.hpp"

namespace PV {

class FeedbackConn : public TransposeConn {
  public:
   FeedbackConn(char const *name, PVParams *params, Communicator const *comm);

   virtual ~FeedbackConn();

  protected:
   FeedbackConn();

   void initialize(char const *name, PVParams *params, Communicator const *comm);

   virtual ConnectionData *createConnectionData() override;
}; // class FeedbackConn

} // namespace PV

#endif // FEEDBACKCONN_HPP_
