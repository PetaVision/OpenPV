/*
 * OjaOjaSTDPConn.cpp
 *
 *  Created on: Sep 27, 2012
 *      Author: dpaiton et slundquist
 */

#include "OjaSTDPConn.hpp"
#include "../layers/LIF.hpp"
#include "../io/io.h"
#include <assert.h>
#include <math.h>

namespace PV {

OjaSTDPConn::OjaSTDPConn(const char * name, HyPerCol * hc, HyPerLayer * pre,
      HyPerLayer * post, const char * filename, InitWeights *weightInit)
{
   initialize_base();
   initialize(name, hc, pre, post, filename, weightInit);
#ifdef PV_USE_OPENCL
   if(gpuAccelerateFlag)
      initializeGPU();
#endif
}

OjaSTDPConn::~OjaSTDPConn()
{
   free(ampLTD); ampLTD = NULL;
   Communicator::freeDatatypes(mpi_datatype); mpi_datatype = NULL;
   deleteWeights();
}

int OjaSTDPConn::deleteWeights()
{
   if (!plasticityFlag) return PV_SUCCESS;

   pvcube_delete(post_stdp_tr);
   pvcube_delete(post_oja_tr);
   pvcube_delete(post_int_tr);

   post_stdp_tr = NULL;
   post_oja_tr  = NULL;
   post_int_tr  = NULL;

   for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
      pvcube_delete(pre_stdp_tr[arborID]);
      pvcube_delete(pre_oja_tr[arborID]);
      pre_stdp_tr[arborID]  = NULL;
      pre_oja_tr[arborID]   = NULL;
   }

   return PV_SUCCESS; //HyPerConn destructor is automatically called
}


int OjaSTDPConn::initialize_base() {
   // Default STDP parameters for modifying weights; defaults are overridden in setParams().
   this->post_stdp_tr     = NULL;
   this->post_oja_tr      = NULL;
   this->post_int_tr      = NULL;
   this->pre_stdp_tr      = NULL;
   this->pre_oja_tr       = NULL;
   this->mpi_datatype     = NULL;

   this->ampLTP           = 1;
   this->ampLTD           = NULL; // Will allocate later
   this->initAmpLTD       = 1;
   this->targetPostRateHz = 1;
   this->targetPreRateHz  = 20;
   this->LTDscale         = ampLTP;
   this->weightDecay      = 0.01;
   this->dWMax            = 1;
   this->weightScale      = 0.25;

   this->tauLTP           = 16.8;
   this->tauLTD           = 33.7;
   this->tauOja           = 337;
   this->tauTHR           = 1000;
   this->tauO             = 1/targetPostRateHz;

   this->wMin             = 0.0001f;
   this->wMax             = 1;

   this->ojaFlag          = true;
   this->synscalingFlag   = false;
   this->synscaling_v     = 1;

   return PV_SUCCESS;
}

int OjaSTDPConn::initialize(const char * name, HyPerCol * hc,
      HyPerLayer * pre, HyPerLayer * post,
      const char * filename, InitWeights *weightInit)
{

   int status = HyPerConn::initialize(name, hc, pre, post, filename, weightInit);

   status |= initPlasticityPatches(); // needs to be called after HyPerConn::initialize since it depends on post being set

   point2PreSynapticWeights(); //set up post synaptic weight monitoring

   //Need to assert that the previous layer is a LIF layer.
   HyPerLayer * postHyPerLayer = this->postSynapticLayer();
   LIF * postLIF = NULL;
   postLIF = dynamic_cast <LIF*> (postHyPerLayer);
   assert(postLIF != NULL);

   //Grab VthRest from presynaptic LIF params
   float VthRest;
   VthRest = postLIF->getLIFParams()->VthRest;

   //allocate ampLTD and set to initial value
   //Restricted post
   ampLTD = (float *) calloc(post->getNumNeurons(), sizeof(float));
   for (int kRes = 0; kRes < post->getNumNeurons(); kRes++) {
      ampLTD[kRes] = initAmpLTD;
   }

   //set LTDscale param (should default to ampLTP
   LTDscale = hc->parameters()->value(name, "LTDscale", LTDscale);
   if (LTDscale < 0) {
      if (hc->columnId()==0) {
         fprintf(stderr,"OjaSTDPConn \"%s\": LTDscale must be positive (value in params is %f).\n", name, LTDscale);
      }
      abort();
   }

   mpi_datatype = Communicator::newDatatypes(pre->getLayerLoc());
   if (mpi_datatype==NULL) {
      fprintf(stderr, "LCALIFLateralKernelConn \"%s\" error creating mpi_datatype\n", name);
      abort();
   }

   return status;
}

// set member variables specified by user
/*
 * Using a dynamic_cast operator to convert (downcast) a pointer to a base class (HyPerLayer)
 * to a pointer to a derived class (LIF). This way I do not need to define a virtual
 * function getWmax() in HyPerLayer which only returns a NULL pointer in the base class.
 */
int OjaSTDPConn::setParams(PVParams * params)
{
   HyPerConn::setParams(params);

   ampLTP           = params->value(getName(), "ampLTP", ampLTP);
   initAmpLTD       = params->value(getName(), "initAmpLTD", initAmpLTD);
   tauLTP           = params->value(getName(), "tauLTP", tauLTP);
   tauLTD           = params->value(getName(), "tauLTD", tauLTD);
   tauOja           = params->value(getName(), "tauOja", tauOja);
   tauTHR           = params->value(getName(), "tauTHR", tauTHR);
   tauO             = params->value(getName(), "tauO",tauO);

   weightDecay      = params->value(getName(), "weightDecay", weightDecay);
   targetPostRateHz = params->value(getName(), "targetPostRate", targetPostRateHz);
   targetPreRateHz  = params->value(getName(), "targetPreRate", targetPreRateHz);

   ojaFlag          = params->value(getName(), "ojaFlag",ojaFlag);
   synscalingFlag   = params->value(getName(), "synscalingFlag", synscalingFlag);
   synscaling_v     = params->value(getName(), "synscaling_v", synscaling_v);

   if (!ojaFlag) { //Don't even look for the param if ojaFlag is set to 1
      wMax          = params->value(getName(), "wMax", wMax);
   }
   wMin             = params->value(getName(), "wMin", wMin);
   // dWMax            = params->value(getName(), "dWMax", dWMax); // dWMax is set in HyPerConn::setParams, called above
   weightScale      = params->value(getName(), "weightScale", weightScale);

   return 0;
}

int OjaSTDPConn::initPlasticityPatches()
{
   if (!plasticityFlag) return PV_SUCCESS;

   int status = HyPerConn::initPlasticityPatches();
   assert(status == 0);

   post_stdp_tr = pvcube_new(&post->getCLayer()->loc, post->getNumNeurons());
   post_oja_tr  = pvcube_new(&post->getCLayer()->loc, post->getNumNeurons());
   post_int_tr  = pvcube_new(&post->getCLayer()->loc, post->getNumNeurons());

   //Pre traces for each arbor delay
   pre_stdp_tr = (PVLayerCube **) calloc(numberOfAxonalArborLists(), sizeof(PVLayerCube *));
   pre_oja_tr  = (PVLayerCube **) calloc(numberOfAxonalArborLists(), sizeof(PVLayerCube *));

   assert(post_stdp_tr != NULL);
   assert(post_oja_tr  != NULL);
   assert(post_int_tr  != NULL);

   const float targetPostRatekHz = targetPostRateHz/1000; // Convert Hz to kHz

   for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
      pre_stdp_tr[arborID] = pvcube_new(&pre->getCLayer()->loc, pre->getNumExtended());
      pre_oja_tr[arborID]  = pvcube_new(&pre->getCLayer()->loc, pre->getNumExtended());

      assert(pre_stdp_tr[arborID] != NULL);
      assert(pre_oja_tr[arborID]  != NULL);
   }

   int nkPost = post_stdp_tr->numItems;
   assert(nkPost == post->getNumNeurons());
   for (int kPostRes = 0; kPostRes < nkPost; kPostRes++) {
      post_stdp_tr->data[kPostRes] = tauLTD * targetPostRatekHz;
      post_oja_tr->data[kPostRes]  = tauOja * targetPostRatekHz;
      post_int_tr->data[kPostRes]  = tauO   * targetPostRatekHz;
   }
   for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
      int numPre  = pre_stdp_tr[arborID]->numItems; //should be extended
      for (int kPreExt = 0; kPreExt < numPre; kPreExt++) {
         pre_stdp_tr[arborID]->data[kPreExt] = tauLTP * targetPostRatekHz;
         pre_oja_tr[arborID]->data[kPreExt]  = tauOja * targetPostRatekHz;
      }
   }

   return PV_SUCCESS;
}

#ifdef PV_USE_OPENCL
int OjaSTDPConn::initializeThreadBuffers(const char * kernelName) {return 0;}
int OjaSTDPConn::initializeThreadKernels(const char * kernelName) {return 0;}
#endif // PV_USE_OPENCL

/**
 * First function to be executed
 * Updates the postsynaptic trace and calls the updateWeights function
 */
int OjaSTDPConn::updateState(double time, double dt)
{
   //HyPerConn updateState() does not need to be called because it does not do anything.
   update_timer->start();

   int status=0;
   if (plasticityFlag) {
      status=updateAmpLTD();
      for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
         status=updateWeights(arborID);
      }
   }

   if(synscalingFlag){
      scaleWeights();
   }

   update_timer->stop();

   return status;
}

int OjaSTDPConn::updateAmpLTD()
{
   // Steps:
   // 1. Update post_stdp_tr

   const float dt                = parent->getDeltaTime();
   const float decayLTD          = exp(-dt / tauLTD);
   const float decayOja          = exp(-dt / tauOja);
   const float decayO            = exp(-dt / tauO);
   const float targetPostRatekHz = targetPostRateHz/1000; // Convert Hz to kHz

   //Restricted Post
   const int nkPost = post_stdp_tr->numItems;
   assert(nkPost == post->getNumNeurons());

   //Extended Post
   const pvdata_t * aPost        = post->getLayerData();

   //Restricted Post
   pvdata_t * post_stdp_tr_m;   // Postsynaptic trace matrix; i.e. data of post_stdp_tr struct
   pvdata_t * post_oja_tr_m;    // Postsynaptic mean trace matrix
   pvdata_t * post_int_tr_m;    // Postsynaptic mean trace matrix

   post_stdp_tr_m = post_stdp_tr->data;
   post_oja_tr_m  = post_oja_tr->data;
   post_int_tr_m  = post_int_tr->data;

   //Restricted post vals
   const int postNx = post->getLayerLoc()->nx;
   const int postNy = post->getLayerLoc()->ny;
   const int postNf = post->getLayerLoc()->nf;
   const int postNb = post->getLayerLoc()->nb;

   // 1. Updates the postsynaptic traces
   for (int kPostRes = 0; kPostRes < nkPost; kPostRes++)
   {
      int kPostExt = kIndexExtended(kPostRes, postNx, postNy, postNf, postNb);
      post_stdp_tr_m[kPostRes] = decayLTD * (post_stdp_tr_m[kPostRes] + aPost[kPostExt]);
      post_oja_tr_m[kPostRes]  = decayOja * (post_oja_tr_m[kPostRes]  + aPost[kPostExt]);
      post_int_tr_m[kPostRes]  = decayO   * (post_int_tr_m[kPostRes]  + aPost[kPostExt]);

      ampLTD[kPostRes] += (dt/tauTHR) * ((post_int_tr_m[kPostRes]/tauO) - targetPostRatekHz) * (LTDscale/targetPostRatekHz);
      ampLTD[kPostRes]  = ampLTD[kPostRes] < 0 ? 0 : ampLTD[kPostRes]; // ampLTD should not go below 0
      assert(ampLTD[kPostRes] == ampLTD[kPostRes]); // Make sure it is not NaN
   }
   return 0;
}

int OjaSTDPConn::updateWeights(int arborID)
{
   // Steps:
   // 2. Update pre_stdp_tr[arborID]
   // 3. Update w_ij

   const float dt                = parent->getDeltaTime();
   const float decayLTP          = exp(-dt / tauLTP);
   const float decayOja          = exp(-dt / tauOja);
   const float targetPostRatekHz = targetPostRateHz/1000; // Convert Hz to kHz
   const float targetPreRatekHz  = targetPreRateHz/1000; // Convert Hz to kHz

   //Extended Pre
   const int nkPre = pre->getNumExtended();
   assert(nkPre == getNumWeightPatches());


   //Restricted Post
   const pvdata_t * aPost = post->getLayerData(getDelay(arborID));
   pvdata_t * post_stdp_tr_m;   // Postsynaptic trace matrix; i.e. data of post_stdp_tr struct
   pvdata_t * post_oja_tr_m;    // Postsynaptic mean trace matrix
   pvdata_t * ampLTD_m;         // local ampLTD

   //Extended Pre
   const pvdata_t * preLayerData = pre->getLayerData(getDelay(arborID));
   pvdata_t aPre;
   pvdata_t * pre_stdp_tr_m;    // Presynaptic trace matrix
   pvdata_t * pre_oja_tr_m;
   pvdata_t * W;                // Weight matrix pointer

   //Restricted post vals
   const int postNx = post->getLayerLoc()->nx;
   const int postNy = post->getLayerLoc()->ny;
   const int postNf = post->getLayerLoc()->nf;
   const int postNb = post->getLayerLoc()->nb;

   //stride in restricted space
   const int postStrideYRes = postNf * postNx;

   float scaleFactor;
   if (ojaFlag) {
      scaleFactor = dWMax * powf(dt/tauOja,2.0);
   }
   else
   {
      scaleFactor = dWMax;
   }
   assert(scaleFactor == scaleFactor); // Make sure it is not NaN (can only happen if tauOja or targetPostRatekHz = 0)

   int nk, ny;

#ifdef SPLIT_PRE_POST //Separate LTD and LTP calculations to take advantage of sparsity
   //Loop over postsynaptic neurons for LTP
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Number of pre-neurons in post receptive field. Postsynaptic weights are never shrunken

   std::cout << "\nnew\n";

   //Update all pre traces (all traces decay every time step)
   for (int kPreExt = 0; kPreExt < nkPre; kPreExt++)           // Loop over all presynaptic neurons
   {
      aPre           = preLayerData[kPreExt];                  // Spiking activity
      pre_stdp_tr_m  = &(pre_stdp_tr[arborID]->data[kPreExt]); // PreTrace for given presynaptic neuron kPreExt
      pre_oja_tr_m   = &(pre_oja_tr[arborID]->data[kPreExt]);

      // 2. Updates the presynaptic trace
      *pre_stdp_tr_m = decayLTP * ((*pre_stdp_tr_m) + aPre);        //If spiked, minimum is 1. If no spike, minimum is 0.
      *pre_oja_tr_m  = decayOja * ((*pre_oja_tr_m)  + aPre);
   }

   float * startAdd = this->get_wDataStart(arborID); // Address of first neuron in pre layer
   //Loop through postsynaptic neurons (non-extended indices)
   for (int kPost = 0; kPost < post->getNumNeurons(); kPost++) { //Neuron indices
	   //Post in extended space
	   if (aPost[kPost] == 0) { //No LTP if post does not spike
		   continue;
	   }

	   post_oja_tr_m  = &(post_oja_tr->data[kPost]); //Address of post trace in restricted space
	   pvdata_t ** postData = getPostWeightsp(arborID,kPost); // Pointer array full of addresses pointing to the weights for all of the preNeurons connected to the given postNeuron's receptive field
	   for (int kPrePatch=0; kPrePatch < numPostPatch; kPrePatch++) { // Loop through all pre-neurons connected to given post-neuron
		   float * kPreAdd = postData[kPrePatch];  // Address of preNeuron in receptive field of postNeuron
		   assert(kPreAdd != NULL);

           int kPreExt = (kPreAdd-startAdd) / (this->xPatchSize()*this->yPatchSize()*this->fPatchSize()); // Grab index based on patch size
           assert(kPreExt < nkPre);

		   // Pre in extended space
		   pre_stdp_tr_m  = &(pre_stdp_tr[arborID]->data[kPreExt]); // PreTrace for given presynaptic neuron kPreExt
		   pre_oja_tr_m   = &(pre_oja_tr[arborID]->data[kPreExt]);

		   std::cout << "\nnkPre: "<<nkPre<<" numPostPatch: "<<numPostPatch<<" kPost: "<<kPost<<" kPrePatch: "<<kPrePatch<<" kPreExt: "<<kPreExt<<" trace: "<<(*pre_oja_tr_m)<<"\n";

		   // See STDP_LCA_Equations.pdf in documentation for description of Oja (feed-forward weight adaptation) equations. TODO: That file does not exist.
		   float ojaTerm;
		   if (ojaFlag) {
			   ojaTerm = (*post_oja_tr_m) * ((*pre_oja_tr_m) - (targetPreRatekHz/targetPostRatekHz) * ((*postData[kPrePatch]) / weightScale) * (*post_oja_tr_m));
			   assert(ojaTerm == ojaTerm); // Make sure it is not NaN (only happens if tauOja is 0)
		   } else { //should just be standard STDP at this point
			   ojaTerm = 1.0;
		   }

		   //STDP Equation
           (*postData[kPrePatch]) += scaleFactor * ojaTerm * ampLTP * aPost[kPost] * (*pre_stdp_tr_m);

           (*postData[kPrePatch]) = (*postData[kPrePatch]) < wMin ? wMin : (*postData[kPrePatch]); // Stop weights from going all the way to 0
		   if (!ojaFlag) { //oja term should get rid of the need to impose a maximum weight
			   (*postData[kPrePatch]) = (*postData[kPrePatch]) > wMax ? wMax : (*postData[kPrePatch]);
		   }
	   }
   }

   // Pre-synaptic neurons for LTD
   for (int kPreExt = 0; kPreExt < nkPre; kPreExt++) // Loop over all presynaptic neurons
   {
      //Pre in extended space
      aPre = preLayerData[kPreExt]; // Spiking activity
      if (aPre == 0) { //No LTD if pre does not spike
    	  continue;
      }

      size_t postOffsetExt = getAPostOffset(kPreExt, arborID); // Gets start index for postsynaptic vectors for given presynaptic neuron and axon
      // size_t postOffsetRes = postOffsetExt - (postNb * (postNx + 2*postNb) + postNb);
      size_t postOffsetRes = kIndexRestricted(postOffsetExt, postNx, postNy, postNf, postNb);
      //Post in restricted space
      post_stdp_tr_m = &(post_stdp_tr->data[postOffsetRes]);   // Reference to STDP post trace (local)
      ampLTD_m       = &(ampLTD[postOffsetRes]);               // Points to local address

      W = get_wData(arborID, kPreExt);                         // Pointer to data of given axon & presynaptic neuron

      // Get weights in form of a patch (nx,ny,nf)
      // nk and ny are the number of neurons connected to the given presynaptic neuron in the x*nfp and y
      // if each of the presynaptic neurons connects to all postsynaptic than nk*ny = nkPost TODO: Is this true? Rui says yes.
      PVPatch * w = getWeights(kPreExt, arborID);                // Get weights in form of a patch (nx,ny,nf), TODO: what's the role of the offset?
      nk  = nfp * w->nx; // one line in x at a time
      ny  = w->ny;

      // 3. Update weights
      for (int y = 0; y < ny; y++) {
         for (int kPatchLoc = 0; kPatchLoc < nk; kPatchLoc++) { //loop over all postsynaptic neurons connected to given presynaptic neuron

            //STDP Equation
            W[kPatchLoc] -= scaleFactor * ampLTD_m[kPatchLoc] * aPre * post_stdp_tr_m[kPatchLoc];

            W[kPatchLoc] = W[kPatchLoc] < wMin ? wMin : W[kPatchLoc]; // Stop weights from going all the way to 0
            if (!ojaFlag) { //oja term should get rid of the need to impose a maximum weight
               W[kPatchLoc] = W[kPatchLoc] > wMax ? wMax : W[kPatchLoc];
            }
         }

         // advance pointers in y
         W += syp;

         // postActivity and post trace are extended layer
         post_stdp_tr_m += postStrideYRes;
         ampLTD_m       += postStrideYRes;
      }
   }
#else
   // this stride is in extended space for post-synaptic activity and STDP decrement variable
   const int postStrideYExt = postNf * (postNx + 2 * postNb);

   //std::cout << "\nold\n";

   for (int kPreExt = 0; kPreExt < nkPre; kPreExt++)           // Loop over all presynaptic neurons
   {
      size_t postOffsetExt = getAPostOffset(kPreExt, arborID); // Gets start index for postsynaptic vectors for given presynaptic neuron and axon
      // size_t postOffsetRes = postOffsetExt - (postNb * (postNx + 2*postNb) + postNb);
      size_t postOffsetRes = kIndexRestricted(postOffsetExt, postNx, postNy, postNf, postNb);

      //Post in extended space
      aPost          = &post->getLayerData()[postOffsetExt];   // Gets address of postsynaptic activity
      //Post in restricted space
      post_stdp_tr_m = &(post_stdp_tr->data[postOffsetRes]);   // Reference to STDP post trace (local)
      post_oja_tr_m  = &(post_oja_tr->data[postOffsetRes]);
      ampLTD_m       = &(ampLTD[postOffsetRes]);               // Points to local address
      //Pre in extended space
      aPre           = preLayerData[kPreExt];                  // Spiking activity
      pre_stdp_tr_m  = &(pre_stdp_tr[arborID]->data[kPreExt]); // PreTrace for given presynaptic neuron kPre
      pre_oja_tr_m   = &(pre_oja_tr[arborID]->data[kPreExt]);

      W = get_wData(arborID, kPreExt);                         // Pointer to data of given axon & presynaptic neuron

      // Get weights in form of a patch (nx,ny,nf)
      // nk and ny are the number of neurons connected to the given presynaptic neuron in the x*nfp and y
      // if each of the presynaptic neurons connects to all postsynaptic than nk*ny = nkPost TODO: Is this true? Rui says yes.
      PVPatch * w = getWeights(kPreExt, arborID);                // Get weights in form of a patch (nx,ny,nf), TODO: what's the role of the offset?
      nk  = nfp * w->nx; // one line in x at a time
      ny  = w->ny;

      // 2. Updates the presynaptic trace
      *pre_stdp_tr_m = decayLTP * ((*pre_stdp_tr_m) + aPre);        //If spiked, minimum is 1. If no spike, minimum is 0.
      *pre_oja_tr_m  = decayOja * ((*pre_oja_tr_m)  + aPre);

	   //std::cout << "\n\n"<<kPreExt<<"\t"<<(*pre_oja_tr_m)<<"\n";
      // 3. Update weights
      for (int y = 0; y < ny; y++) {
         for (int kPatchLoc = 0; kPatchLoc < nk; kPatchLoc++) { //loop over all postsynaptic neurons connected to given presynaptic neuron

            // See STDP_LCA_Equations.pdf in documentation for description of Oja (feed-forward weight adaptation) equations. TODO: That file does not exist.
            float ojaTerm;
            if (ojaFlag) {
               ojaTerm = post_oja_tr_m[kPatchLoc] * ((*pre_oja_tr_m) - (targetPreRatekHz/targetPostRatekHz) * (W[kPatchLoc]/weightScale) * post_oja_tr_m[kPatchLoc]);
               assert(ojaTerm == ojaTerm); // Make sure it is not NaN (only happens if tauOja is 0)
            } else { //should just be standard STDP at this point
              ojaTerm = 1.0;
            }

            //STDP Equation
            W[kPatchLoc] += scaleFactor *
              (ojaTerm * ampLTP * aPost[kPatchLoc] * (*pre_stdp_tr_m) - ampLTD_m[kPatchLoc] * aPre * post_stdp_tr_m[kPatchLoc] - weightDecay * W[kPatchLoc]);

            W[kPatchLoc] = W[kPatchLoc] < wMin ? wMin : W[kPatchLoc]; // Stop weights from going all the way to 0
            if (!ojaFlag) { //oja term should get rid of the need to impose a maximum weight
               W[kPatchLoc] = W[kPatchLoc] > wMax ? wMax : W[kPatchLoc];
            }
         }

         // advance pointers in y
         W += syp;

         // postActivity and post trace are extended layer
         aPost          += postStrideYExt; //TODO: is this really in the extended space?
         post_stdp_tr_m += postStrideYRes;
         post_oja_tr_m  += postStrideYRes;
         ampLTD_m       += postStrideYRes;
      }
   }
#endif
   return 0;
}

int OjaSTDPConn::scaleWeights() {
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken

   float sumW = 0;
   const int xScale = post->getXScale() - pre->getXScale();
   const int yScale = post->getYScale() - pre->getYScale();
   const double powXScale = pow(2.0f, (double) xScale);
   const double powYScale = pow(2.0f, (double) yScale);

   nxpPost = (int) (nxp * powXScale);
   nypPost = (int) (nyp * powYScale);
   nfpPost = pre->clayer->loc.nf;

   for(int arborID=0;arborID<numberOfAxonalArborLists();arborID++) {
      //Loop through post-synaptic neurons (non-extended indices)
      for (int kPost = 0; kPost < post->getNumNeurons(); kPost++) {

         pvdata_t ** postData = wPostDataStartp[arborID] + numPostPatch*kPost + 0;
         for (int kp = 0; kp < numPostPatch; kp++) { //TODO: Scale only the weights non-extended space
            sumW += *(postData[kp]);
         }
         for (int kp = 0; kp < numPostPatch; kp++) {
            *(postData[kp]) = ((*postData[kp])/sumW)*synscaling_v;
         }
         sumW = 0;
      }
   }
   return PV_SUCCESS;
}

pvdata_t ** OjaSTDPConn::getPostWeightsp(int arborID, int kPost) {
   const int numPostPatch = nxpPost * nypPost * nfpPost; // Post-synaptic weights are never shrunken
   pvdata_t ** postData = wPostDataStartp[arborID] + numPostPatch*kPost + 0;

   return postData;
}

int OjaSTDPConn::outputState(double timef, bool last)
{
   int status;

   if (last) {
      printf("Writing last STDP weights..%f\n",timef);
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, last);
      assert(status == 0);
   }else if ( (timef >= writeTime) && (writeStep >= 0) ) {
      //writeTime += writeStep; Done in HyperConn
      convertPreSynapticWeights(timef);
      status = writePostSynapticWeights(timef, false);
      assert(status == 0);

      // append to output file after original open
      //ioAppend = true;
   }

   status = HyPerConn::outputState(timef, last);


   if (status != PV_SUCCESS) return status;


   return status;
}

// Overwrite from HyPerConn to return the max limit, instead of computing the max weight in all weights
float OjaSTDPConn::maxWeight(int arborID)
{
   return wMax;
}

// Anything extra that you want to write out when HyPerConn calls writeTextWeights
int OjaSTDPConn::writeTextWeightsExtra(FILE * fd, int k, int arborID)
{
   if (plasticityFlag) {
      pv_text_write_patch(fd, getWeights(k, arborID), get_dwData(arborID, k), nfp, sxp, syp, sfp); // write data[xp,yp,fp]
   }
   return 0;
}

// Change to checkpointWrite and checkpointRead, Jan 10, 2013.
// The presynaptic traces have size nx-by-ny with border of width nb.
// Before the change, the presynaptic traces were checkpointed as if the
// local layer had size (nx+2*nb)-by-(ny+2*nb) with border region zero.
// This has the drawback that the file depends on the MPI configuration.
//
// As of this update, the layer is treated as nx-by-ny with border nb.
// On reading from a checkpoint, the border region is filled by calling
// Communicator::exchange and (if mirrorBCFlag is set), mirroring to the
// exterior boundary.  This should make the trace files independent of
// the MPI configuration, and should recover the border regions correctly.
int OjaSTDPConn::checkpointWrite(const char * cpDir) {
   int status = HyPerConn::checkpointWrite(cpDir);
   char filename[PV_PATH_MAX];
   int chars_needed;
   const PVLayerLoc * loc;

   // **** PRE LAYER INFO *** //
   loc = pre->getLayerLoc();
// Commented out Jan 10, 2013
//    memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
//   // This is kind of hacky, but we save the extended buffers pre_stdp_tr as if they were nonextended buffers of size (nx+2*nb)-by-(ny+2*nb)-by-nf
//   // post_stdp_tr is buffer of size nx-by-ny-by-nf
//   loc.nx += 2*loc.nb;
//   loc.ny += 2*loc.nb;
//   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
//   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
//   loc.nb = 0;

   pvdata_t ** traces = (pvdata_t **) calloc(numberOfAxonalArborLists(), sizeof(pvdata_t *));
   // pre_stdp_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointWrite error.  Path \"%s/%s_pre_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   for (int arborID=0; arborID<numberOfAxonalArborLists(); arborID++) {
      traces[arborID] = pre_stdp_tr[arborID]->data;
   }
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), traces, numberOfAxonalArborLists(), /*extended*/ false, loc);
   // pre_oja_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_oja_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointWrite error.  Path \"%s/%s_pre_oja_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
      traces[arborID] = pre_oja_tr[arborID]->data;
   }
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), traces, numberOfAxonalArborLists(), /*extended*/ false, loc);

   free(traces); traces = NULL;

   // **** POST LAYER INFO *** //
   loc = post->getLayerLoc();
//   memcpy(&loc, post->getLayerLoc(), sizeof(PVLayerLoc));

   // post_stdp_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointWrite error.  Path \"%s/%s_post_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), &post_stdp_tr->data, /*numbands*/ 1, /*extended*/ false, loc);

   // post_oja_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_oja_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointWrite error.  Path \"%s/%s_post_oja_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), &post_oja_tr->data, /*numbands*/ 1, /*extended*/ false, loc);

   // post_int_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_int_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointWrite error.  Path \"%s/%s_post_int_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), &post_int_tr->data, /*numbands*/ 1, /*extended*/ false, loc);

   // ampLTD
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_ampLTD.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointWrite error.  Path \"%s/%sampLTD.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::writeBufferFile(filename, parent->icCommunicator(), (double) parent->simulationTime(), &ampLTD, /*numbands*/ 1, /*extended*/ false, loc);

   return status;
}

int OjaSTDPConn::checkpointRead(const char * cpDir, double * timef) {
   int status = HyPerConn::checkpointRead(cpDir, timef);
   char filename[PV_PATH_MAX];
   int chars_needed;
   double timed;
   const PVLayerLoc * loc;

   // **** PRE LAYER INFO *** //
   loc = pre->getLayerLoc();
// Commented out Jan 10, 2013
//   memcpy(&loc, pre->getLayerLoc(), sizeof(PVLayerLoc));
//   loc.nx += 2*loc.nb;
//   loc.ny += 2*loc.nb;
//   loc.nxGlobal = loc.nx * parent->icCommunicator()->numCommColumns();
//   loc.nyGlobal = loc.ny * parent->icCommunicator()->numCommRows();
//   loc.nb = 0;


   //TODO: Only read if plasticity flag is on (How much of the code below does this todo apply to?)
   pvdata_t ** traces = (pvdata_t **) calloc(numberOfAxonalArborLists(), sizeof(pvdata_t *));
   // pre_stdp_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointRead error.  Path \"%s/%s_pre_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }

   for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
      traces[arborID] = pre_stdp_tr[arborID]->data;
   }
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, traces, numberOfAxonalArborLists(), /*extended*/ true, loc);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning in OjaSTDPConn: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }
   // Exchange borders
   for (int arborID=0; arborID<numberOfAxonalArborLists(); arborID++) {
      if ( pre->useMirrorBCs() ) {
         for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
            pre->mirrorInteriorToBorder(borderId, pre_stdp_tr[arborID], pre_stdp_tr[arborID]);
         }
      }
      parent->icCommunicator()->exchange(pre_stdp_tr[arborID]->data, mpi_datatype, loc);
   }

   // pre_oja_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_pre_oja_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointRead error.  Path \"%s/%s_pre_oja_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   for(int arborID = 0; arborID<numberOfAxonalArborLists(); arborID++) {
      traces[arborID] = pre_oja_tr[arborID]->data;
   }
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, traces, numberOfAxonalArborLists(), /*extended*/ false, loc);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning in OjaSTDPConn: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }
   // Exchange borders
   for (int arborID=0; arborID<numberOfAxonalArborLists(); arborID++) {
      if ( pre->useMirrorBCs() ) {
         for (int borderId = 1; borderId < NUM_NEIGHBORHOOD; borderId++){
            pre->mirrorInteriorToBorder(borderId, pre_oja_tr[arborID], pre_oja_tr[arborID]);
         }
      }
      parent->icCommunicator()->exchange(pre_oja_tr[arborID]->data, mpi_datatype, loc);
   }

   free(traces); traces=NULL;

   // **** POST LAYER INFO *** //
   loc = post->getLayerLoc();
//   memcpy(&loc, post->getLayerLoc(), sizeof(PVLayerLoc));

   // post_stdp_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_stdp_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointRead error.  Path \"%s/%s_post_stdp_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &post_stdp_tr->data, /*numbands*/1, /*extended*/ false, loc);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning in OjaSTDPConn: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }


   // post_oja_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_oja_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointRead error.  Path \"%s/%s_post_oja_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &post_oja_tr->data, /*numbands*/1, /*extended*/ false, loc);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning in OjaSTDPConn: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   // post_int_tr
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_post_int_tr.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointRead error.  Path \"%s/%s_post_int_tr.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &post_int_tr->data, /*numbands*/1, /*extended*/ false, loc);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning in OjaSTDPConn: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   // ampLTD
   chars_needed = snprintf(filename, PV_PATH_MAX, "%s/%s_ampLTD.pvp", cpDir, name);
   if (chars_needed >= PV_PATH_MAX) {
      fprintf(stderr, "OjaSTDPConn::checkpointRead error.  Path \"%s/%sampLTD.pvp\" is too long.\n", cpDir, name);
      abort();
   }
   HyPerLayer::readBufferFile(filename, parent->icCommunicator(), &timed, &ampLTD, /*numbands*/1, /*extended*/ false, loc);
   if( (float) timed != *timef && parent->icCommunicator()->commRank() == 0 ) {
      fprintf(stderr, "Warning: %s and %s_A.pvp have different timestamps: %f versus %f\n", filename, name, (float) timed, *timef);
   }

   return status;
}

} // End of namespace PV
