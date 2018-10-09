#ifndef BINNINGLAYER_HPP_
#define BINNINGLAYER_HPP_

#include "HyPerLayer.hpp"
#include "components/OriginalLayerNameParam.hpp"

namespace PV {

class BinningLayer : public PV::HyPerLayer {
  public:
   BinningLayer(const char *name, HyPerCol *hc);
   virtual Response::Status allocateDataStructures() override;
   virtual bool activityIsSpiking() override { return false; }
   virtual ~BinningLayer();

  protected:
   BinningLayer();
   int initialize(const char *name, HyPerCol *hc);
   virtual void createComponentTable(char const *description) override;
   virtual LayerInputBuffer *createLayerInput() override;
   virtual InternalStateBuffer *createInternalState() override;
   virtual OriginalLayerNameParam *createOriginalLayerNameParam();
   int ioParamsFillGroup(enum ParamsIOFlag ioFlag) override;
   void ioParam_binMaxMin(enum ParamsIOFlag ioFlag);
   void ioParam_delay(enum ParamsIOFlag ioFlag);
   void ioParam_binSigma(enum ParamsIOFlag ioFlag);
   void ioParam_zeroNeg(enum ParamsIOFlag ioFlag);
   void ioParam_zeroDCR(enum ParamsIOFlag ioFlag);
   void ioParam_normalDist(enum ParamsIOFlag ioFlag);
   virtual Response::Status
   communicateInitInfo(std::shared_ptr<CommunicateInitInfoMessage const> message) override;
   void setOriginalLayer();
   virtual void initializeActivity() override;
   virtual Response::Status updateState(double timef, double dt) override;
   void doUpdateState(
         double timed,
         double dt,
         const PVLayerLoc *origLoc,
         const PVLayerLoc *currLoc,
         const float *origData,
         float *currV,
         float binMax,
         float binMin);

   float getSigma() { return binSigma; }
   float calcNormDist(float xVal, float mean, float binSigma);

  private:
   int initialize_base();
   int delay;
   float binMax;
   float binMin;
   float binSigma;
   bool zeroNeg;
   bool zeroDCR;
   bool normalDist;

  protected:
   HyPerLayer *mOriginalLayer = nullptr;
}; // class BinningLayer

} /* namespace PV */
#endif
