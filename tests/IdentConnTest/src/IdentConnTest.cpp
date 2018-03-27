//  A program to test IdentConn. It uses input/IdentConnTest.params
//  There are four presynaptic HyPerLayers connected to a postsynaptic HyPerLayer
//  by IdentConns, two on channel 0 and two on channel 1.
//  The result should be that each channel on the post is the sum of the two HyPerLayers
//  connected on that channel.

//  The layers are initialized as follows, according to their *global* index:
//  PreLayerExc0: 0, 1, 4, 9, 16, etc.
//  PreLayerExc0: 1, 3, 5, 7, 9, etc.
//  PreLayerInh0: 0, 0, 2, 4, 8, 12, 18, etc. (half the square of the index, rounded down)
//  PreLayerInh1: 0, 1, 1, 2, 2, 3, 3, etc. (half the index, rounded up)

//  The postsynaptic excitatory channel should end up with 1, 4, 9, 16, etc.
//  The postsynaptic inhibitary channel should end up with 0, 1, 3, 6, 10, etc.

#include <columns/HyPerCol.hpp>
#include <columns/PV_Init.hpp>
#include <connections/IdentConn.hpp>
#include <layers/HyPerLayer.hpp>
#include <utils/PVLog.hpp>

PV::HyPerLayer *findLayer(PV::HyPerCol &hc, std::string const &layerName);
PV::IdentConn *findIdentConn(PV::HyPerCol &hc, std::string const &connName);
void setMargins(PV::HyPerLayer *layer, int const xMargin, int const yMargin);

int main(int argc, char *argv[]) {
   int status = PV_SUCCESS;

   // Initial setup
   PV::PV_Init pv_init(&argc, &argv, false);
   PV::HyPerCol hc(&pv_init);

   PV::HyPerLayer *preLayerExc0 = findLayer(hc, std::string("PreLayerExc0"));
   PV::HyPerLayer *preLayerExc1 = findLayer(hc, std::string("PreLayerExc1"));
   PV::HyPerLayer *preLayerInh0 = findLayer(hc, std::string("PreLayerInh0"));
   PV::HyPerLayer *preLayerInh1 = findLayer(hc, std::string("PreLayerInh1"));
   PV::HyPerLayer *postLayer    = findLayer(hc, std::string("PostLayer"));
   PV::IdentConn *exc0          = findIdentConn(hc, std::string("Exc0"));
   PV::IdentConn *exc1          = findIdentConn(hc, std::string("Exc1"));
   PV::IdentConn *inh0          = findIdentConn(hc, std::string("Inh0"));
   PV::IdentConn *inh1          = findIdentConn(hc, std::string("Inh1"));

   // Give the pre-layers margins, to test converting from extended to restricted indices as needed
   int xMargin = 3;
   int yMargin = 2; // Make sure we didn't swap an x for y somewhere
   setMargins(preLayerExc0, xMargin, yMargin);
   setMargins(preLayerExc1, xMargin, yMargin);
   setMargins(preLayerInh0, xMargin, yMargin);
   setMargins(preLayerInh1, xMargin, yMargin);
   int marginResult;

   // IdentConn should check that pre and post have the same # of neurons, but let's make sure.
   int const numNeurons = postLayer->getNumNeurons();
   FatalIf(
         preLayerExc0->getNumNeurons() != numNeurons,
         "%s does not have the same number of neurons as %s\n",
         preLayerExc0->getDescription_c(),
         postLayer->getDescription_c());
   FatalIf(
         preLayerExc1->getNumNeurons() != numNeurons,
         "%s does not have the same number of neurons as %s\n",
         preLayerExc1->getDescription_c(),
         postLayer->getDescription_c());
   FatalIf(
         preLayerInh0->getNumNeurons() != numNeurons,
         "%s does not have the same number of neurons as %s\n",
         preLayerInh0->getDescription_c(),
         postLayer->getDescription_c());
   FatalIf(
         preLayerInh1->getNumNeurons() != numNeurons,
         "%s does not have the same number of neurons as %s\n",
         preLayerInh1->getDescription_c(),
         postLayer->getDescription_c());

   hc.run();

   PVLayerLoc const preLoc = *preLayerExc0->getLayerLoc();
   int const nx            = preLoc.nx;
   int const ny            = preLoc.ny;
   int const nf            = preLoc.nf;
   // All pre layers should have this nx, ny, nf; and halo of lt=3, rt=3, dn=2, up=2

   for (int k = 0; k < numNeurons; k++) {
      int const kGlobal = globalIndexFromLocal(k, preLoc);
      int const kExt    = kIndexExtended(k, nx, ny, nf, xMargin, xMargin, yMargin, yMargin);

      float const observedExc0Value = preLayerExc0->getLayerData(0)[kExt];
      float const correctExc0Value  = (float)(kGlobal * kGlobal);
      if (observedExc0Value != (float)(correctExc0Value)) {
         ErrorLog().printf(
               "Rank %d, restricted neuron %d: expected %f in %s, but observed %f\n",
               pv_init.getWorldRank(),
               k,
               (double)(correctExc0Value),
               preLayerExc0->getDescription_c(),
               (double)observedExc0Value);
         status = PV_FAILURE;
      }

      float const observedExc1Value = preLayerExc1->getLayerData(0)[kExt];
      float const correctExc1Value  = (float)(kGlobal + kGlobal + 1);
      if (observedExc1Value != (float)(correctExc1Value)) {
         ErrorLog().printf(
               "Rank %d, restricted neuron %d: expected %f in %s, but observed %f\n",
               pv_init.getWorldRank(),
               k,
               (double)(correctExc0Value),
               preLayerExc0->getDescription_c(),
               (double)observedExc1Value);
         status = PV_FAILURE;
      }

      float const correctExcSum  = correctExc0Value + correctExc1Value;
      float const observedExcSum = postLayer->getChannel(CHANNEL_EXC)[k];
      if (observedExcSum != correctExcSum) {
         ErrorLog().printf(
               "Rank %d, restricted neuron %d: expected %f but observed %f\n",
               pv_init.getWorldRank(),
               k,
               (double)correctExcSum,
               (double)observedExcSum);
         status = PV_FAILURE;
      }

      float const observedInh0Value = preLayerInh0->getLayerData(0)[kExt];
      float const correctInh0Value  = (float)(kGlobal * kGlobal / 2 /* integer division */);
      if (observedInh0Value != (float)(correctInh0Value)) {
         ErrorLog().printf(
               "Rank %d, restricted neuron %d: expected %f in %s, but observed %f\n",
               pv_init.getWorldRank(),
               k,
               (double)(correctInh0Value),
               preLayerInh0->getDescription_c(),
               (double)observedInh0Value);
         status = PV_FAILURE;
      }

      float const observedInh1Value = preLayerInh1->getLayerData(0)[kExt];
      float const correctInh1Value  = (float)((kGlobal + 1) / 2 /* integer division */);
      if (observedInh1Value != (float)(correctInh1Value)) {
         ErrorLog().printf(
               "Rank %d, restricted neuron %d: expected %f in %s, but observed %f\n",
               pv_init.getWorldRank(),
               k,
               (double)(correctInh0Value),
               preLayerInh0->getDescription_c(),
               (double)observedInh1Value);
         status = PV_FAILURE;
      }

      float const correctInhSum  = correctInh0Value + correctInh1Value;
      float const observedInhSum = postLayer->getChannel(CHANNEL_INH)[k];
      if (observedInhSum != correctInhSum) {
         ErrorLog().printf(
               "Rank %d, restricted neuron %d: expected %f but observed %f\n",
               pv_init.getWorldRank(),
               k,
               (double)correctInhSum,
               (double)observedInhSum);
         status = PV_FAILURE;
      }
   }

   return status == PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;
}

PV::HyPerLayer *findLayer(PV::HyPerCol &hc, std::string const &layerName) {
   PV::Observer *object = hc.getObjectFromName(layerName);
   FatalIf(
         object == nullptr,
         "%s does not have a layer named \"%s\" in %s\n",
         hc.getDescription_c(),
         layerName.c_str(),
         hc.getPV_InitObj()->getStringArgument(std::string("ParamsFile")).c_str());

   PV::HyPerLayer *layer = dynamic_cast<PV::HyPerLayer *>(object);
   FatalIf(
         layer == nullptr,
         "%s object \"%s\" is not a layer.\n",
         hc.getDescription_c(),
         layerName.c_str());

   return layer;
}

PV::IdentConn *findIdentConn(PV::HyPerCol &hc, std::string const &connName) {
   PV::Observer *object = hc.getObjectFromName(connName);
   FatalIf(
         object == nullptr,
         "%s does not have a layer named \"%s\" in %s\n",
         hc.getDescription_c(),
         connName.c_str(),
         hc.getPV_InitObj()->getStringArgument(std::string("ParamsFile")).c_str());

   PV::IdentConn *conn = dynamic_cast<PV::IdentConn *>(object);
   FatalIf(
         conn == nullptr,
         "%s object \"%s\" is not an IdentConn.\n",
         hc.getDescription_c(),
         connName.c_str());

   return conn;
}

void setMargins(PV::HyPerLayer *layer, int const xMargin, int const yMargin) {
   int marginResult;
   layer->requireMarginWidth(xMargin, &marginResult, 'x');
   FatalIf(
         marginResult != xMargin,
         "Failed to set x-margin for \"%s\".\n",
         layer->getDescription_c());
   layer->requireMarginWidth(yMargin, &marginResult, 'y');
   FatalIf(
         marginResult != yMargin,
         "Failed to set y-margin for \"%s\".\n",
         layer->getDescription_c());
}
