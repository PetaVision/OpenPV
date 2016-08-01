#ifndef _LATERALINTERACTIONCONNGPU_HPP_
#define _LATERALINTERACTIONCONNGPU_HPP_

#include "HyPerConnGPU.hpp"

namespace GPULCA {
class LateralInteractionConnGPU : public HyPerConnGPU {
 public:
  LateralInteractionConnGPU();
  LateralInteractionConnGPU(const char* name, PV::HyPerCol* hc);

  virtual ~LateralInteractionConnGPU();
  virtual int communicateInitInfo();
  virtual int allocateDataStructures();
  virtual int deliver();

  HyPerConnGPU* getOriginalConn() { return originalConn; }

 private:
  char* originalConnName;
  HyPerConnGPU* originalConn;
};
}

#endif  // _LATERALINTERACTIONCONNGPU_HPP_
