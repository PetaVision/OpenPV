#ifndef INTERACTIVECONTEXT_HPP_
#define INTERACTIVECONTEXT_HPP_

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>

namespace PV {

class InteractiveContext {
   public:
      InteractiveContext(std::map<std::string, std::string> args, std::string params);
      ~InteractiveContext();
      void   beginRun();
      double advanceRun(unsigned int steps);
      void   finishRun();
      void   getLayerActivity(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf); 
      void   getLayerState(const char *layerName, std::vector<float> *data,
                  int *nx, int *ny, int *nf); 
      void   setLayerState(const char *layerName, std::vector<float> *data);
   protected:
      HyPerCol *mHC;
      PV_Init  *mPVI;
      int       mArgC;
      char    **mArgV;
};


} /* namespace PV */

#endif
