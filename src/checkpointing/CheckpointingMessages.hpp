#include "observerpattern/BaseMessage.hpp"

namespace PV {

template <typename T> // In practice, T is always Checkpointer.
// Templated to avoid including Checkpointer.hpp in this file.
class RegisterDataMessage : public BaseMessage {
  public:
   RegisterDataMessage(T *dataRegistry) {
      setMessageType("RegisterData");
      mDataRegistry = dataRegistry;
   }
   T *mDataRegistry;
};

template <typename T> // In practice, T is always Checkpointer.
class ReadStateFromCheckpointMessage : public BaseMessage {
  public:
   ReadStateFromCheckpointMessage(T *dataRegistry) {
      setMessageType("ReadStateFromCheckpoint");
      mDataRegistry = dataRegistry;
   }
   T *mDataRegistry;
};

class ProcessCheckpointReadMessage : public BaseMessage {
  public:
   ProcessCheckpointReadMessage(std::string const &directory) : mDirectory(directory) {
      setMessageType("ProcessCheckpointRead");
   }
   std::string mDirectory;
};

class PrepareCheckpointWriteMessage : public BaseMessage {
  public:
   PrepareCheckpointWriteMessage() { setMessageType("PrepareCheckpointWrite"); }
};

class WriteParamsFileMessage : public BaseMessage {
  public:
   WriteParamsFileMessage(std::string const &directory) : mDirectory(directory) {
      setMessageType("WriteParamsFile");
   }
   std::string mDirectory;
};

} // end namespace PV
