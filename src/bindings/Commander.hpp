#ifndef COMMANDER_HPP_
#define COMMANDER_HPP_


#include <mpi.h>
#include <bindings/Interactions.hpp>

namespace PV {

class Commander {
   public:
      Commander(std::map<std::string, std::string> args, std::string params, void (*errFunc)(std::string const));
      ~Commander();

      static const int MPI_TAG = 666;

      enum Command {
         CMD_NONE,
         CMD_BEGIN,
         CMD_ADVANCE,
         CMD_FINISH,
         CMD_GET_ACTIVITY,
         CMD_GET_STATE,
         CMD_SET_STATE,
         CMD_GET_PROBE_VALUES,
         CMD_SET_WEIGHTS
      };

   protected:
      Interactions *mInteractions;

   private:
      enum Buffer {
         BUF_A,
         BUF_V
      };

      void   rootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootRecv(void *buf, int num, MPI_Datatype dtype);
      void   rootSendCmdName(Command cmd, const char *name);
      std::string const nonRootRecvName();
      int    getRank();
      int    getCommSize();
      int    getRow();
      int    getCol();
      int    getBatch();

   public:
      bool   isRoot();
      bool   isFinished();
      void   waitForCommands();
      void   getLayerActivity(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf);
      void   getLayerState(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf);
      void   setLayerState(const char *layerName, std::vector<float> *data);
      void   getProbeValues(const char *probeName, std::vector<double> *data);
      void   getConnectionWeights(const char *connName, std::vector<float> *data,
                  int *nwp, int *nyp, int *nxp, int *nfp);
      void   setConnectionWeights(const char *connName, std::vector<float> *data);
      void   begin();
      void   finish();
      double advance(unsigned int steps);
   private:
      void   getLayerData(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf, Buffer b);

      void   remoteAdvance();
      void   remoteGetLayerData(Buffer b);
      void   remoteSetLayerState();
      void   remoteGetProbeValues();
      void   remoteSetConnectionWeights();

      void   throwError(std::string const err);
      void   (*mErrFunc)(std::string const);


};


} /* namespace PV */




#endif
