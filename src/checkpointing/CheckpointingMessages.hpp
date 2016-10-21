#include "observerpattern/BaseMessage.hpp"

namespace PV {

class ProcessCheckpointReadMessage : public BaseMessage {
  public:
   ProcessCheckpointReadMessage(std::string const &directory) : mDirectory(directory) {
      setMessageType("ProcessCheckpointReadMessage");
   }
   std::string mDirectory;
};

class PrepareCheckpointWriteMessage : public BaseMessage {
  public:
   PrepareCheckpointWriteMessage(std::string const &directory) : mDirectory(directory) {
      setMessageType("ProcessCheckpointWriteMessage");
   }
   std::string mDirectory;
};

} // end namespace PV
