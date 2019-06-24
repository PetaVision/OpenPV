#ifndef INTERACTIVECONTEXT_HPP_
#define INTERACTIVECONTEXT_HPP_

#define IC_MPI_TAG 666

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>

#include <mpi.h>

namespace PV {

class InteractiveContext {
   public:
      InteractiveContext(std::map<std::string, std::string> args, std::string params);
      ~InteractiveContext();
      void   beginRun();
      double advanceRun(unsigned int steps);
      void   finishRun();
      void   getLayerActivity(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf, int *nb); 
      void   getLayerState(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf, int *nb); 
      void   setLayerState(const char *layerName, std::vector<float> *data);
      void   getLayerShape(const char *layerName, PVLayerLoc *loc);
      bool   isFinished();
      void   getEnergy(const char *probeName, std::vector<double> *data);

      int    getMPIRank();
      int    getMPICommSize();

      void   handleMPI();

   private:

      enum Buffer {
         BUF_A,
         BUF_V
      };

      // TODO: ifdef all MPI stuff
      void   rootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootSend(const void *buf, int num, MPI_Datatype dtype);
      void   nonRootRecv(void *buf, int num, MPI_Datatype dtype);

      void   remoteGetLayerActivity(const char *layerName);
      void   message(std::shared_ptr<BaseMessage const> message);
      void   getLayerData(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf, int *nb, Buffer b);

   protected:
      HyPerCol *mHC;
      PV_Init  *mPVI;
      int       mArgC;
      char    **mArgV;

      enum Command {
         CMD_NONE,
         CMD_ADVANCE_RUN,
         CMD_GET_ACTIVITY
      };


};


} /* namespace PV */

#endif
