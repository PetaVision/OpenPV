#include "io/FileManager.hpp"
#include "observerpattern/BaseMessage.hpp"
#include <memory>

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
   ProcessCheckpointReadMessage(std::shared_ptr<FileManager const> fileManager) :
         mFileManager(fileManager) {
      setMessageType("ProcessCheckpointRead");
   }
   std::shared_ptr<FileManager const> mFileManager;
};

class PrepareCheckpointWriteMessage : public BaseMessage {
  public:
   PrepareCheckpointWriteMessage() { setMessageType("PrepareCheckpointWrite"); }
};

class WriteParamsFileMessage : public BaseMessage {
  public:
   enum Action { WRITE, DELETE };
   WriteParamsFileMessage(
         std::shared_ptr<FileManager const> fileManager,
         std::string const &path,
         Action action) :
         mFileManager(fileManager), mParamsFilePath(path), mAction(action) {
      setMessageType("WriteParamsFile");
   }
   std::shared_ptr<FileManager const> mFileManager;
   std::string mParamsFilePath;
   Action mAction;
};

} // end namespace PV
