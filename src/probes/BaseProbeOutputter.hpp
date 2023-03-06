#ifndef BASEPROBEOUTPUTTER_HPP_
#define BASEPROBEOUTPUTTER_HPP_

#include "checkpointing/Checkpointer.hpp"
#include "columns/Communicator.hpp"
#include "io/FileStream.hpp"
#include "io/PVParams.hpp"
#include "io/PrintStream.hpp"
#include "probes/ProbeComponent.hpp"
#include "structures/MPIBlock.hpp"

#include <memory>
#include <string>
#include <vector>

namespace PV {

class BaseProbeOutputter : public ProbeComponent {
  protected:
   virtual void ioParam_textOutputFlag(enum ParamsIOFlag ioFlag);
   virtual void ioParam_probeOutputFile(enum ParamsIOFlag ioFlag);
   virtual void ioParam_message(enum ParamsIOFlag ioFlag);

  public:
   BaseProbeOutputter(char const *objName, PVParams *params, Communicator const *comm);
   virtual ~BaseProbeOutputter();

   void initOutputStreams(Checkpointer *checkpointer, int localNBatch);
   virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);

   virtual void printHeader() {}

   /**
    * Returns true if a probeOutputFile is being used.
    * Otherwise, returns false (indicating output is going to getOutputStream().
    */
   bool isWritingToFile() const { return mProbeOutputFilename and mProbeOutputFilename[0]; }

   /**
    * Writes the indicated string to the output stream associated with the given batchIndex.
    * The string is prepended by the Message, and appended with a linefeed ("\n").
    */
   void writeString(std::string const &str, int batchIndex);

  protected:
   BaseProbeOutputter() {}
   int calcGlobalBatchOffset() const;
   void initialize(char const *objName, PVParams *params, Communicator const *comm);
   void initMessageString();

   std::shared_ptr<PrintStream> returnOutputStream(int b);

   void printStringToAll(char const *str);
   void printStringToAll(std::string const &str);

   Communicator const *getCommunicator() const { return mCommunicator; }
   std::shared_ptr<MPIBlock const> getIOMPIBlock() const { return mIOMPIBlock; }
   int getLocalNBatch() const { return mLocalNBatch; }
   std::string const &getMessage() const { return mMessageString; }
   char const *getMessageParam() const { return mMessageParam; }
   char const *getProbeOutputFilename() const { return mProbeOutputFilename; }
   bool getTextOutputFlag() const { return mTextOutputFlag; }

  private:
   Communicator const *mCommunicator;
   std::shared_ptr<MPIBlock const> mIOMPIBlock;
   int mLocalNBatch;
   char *mMessageParam = nullptr; // the message parameter in the params
   std::string mMessageString; // the string that gets printed when outputting the stats:
                               // the empty string if message is empty or null;
                               // message + ":" if message is nonempty
   std::vector<std::shared_ptr<FileStream>> mOutputStreams;
   char *mProbeOutputFilename = nullptr;
   bool mTextOutputFlag       = true;

}; // class BaseProbeOutputter

} // namespace PV

#endif // BASEPROBEOUTPUTTER_HPP_
