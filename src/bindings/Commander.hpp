#ifndef COMMANDER_HPP_
#define COMMANDER_HPP_


#include <mpi.h>
#include <bindings/InteractiveContext.hpp>

namespace PV {

class Commander {
   public:
      Commander(std::map<std::string, std::string> args, std::string params);
      ~Commander();

      static const int MPI_TAG = 666;

      enum Command {
         CMD_NONE,
         CMD_BEGIN_RUN,
         CMD_ADVANCE_RUN,
         CMD_FINISH_RUN,
         CMD_GET_ACTIVITY,
         CMD_GET_STATE,
         CMD_SET_STATE,
         CMD_GET_PROBE_VALUES
      };

   protected:
      InteractiveContext *mIC;

   private:
      enum Buffer {
         BUF_A,
         BUF_V
      };

      void   rootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootRecv(void *buf, int num, MPI_Datatype dtype);
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
      void   beginRun();
      void   finishRun();
      double advanceRun(unsigned int steps);
   private:
      void   getLayerData(const char *layerName, std::vector<float> *data,
                  int *nb, int *ny, int *nx, int *nf, Buffer b);
      void   remoteAdvanceRun();
      void   remoteGetLayerData(Buffer b);
      void   remoteSetLayerState();
      void   remoteGetProbeValues();


};


} /* namespace PV */




#endif
