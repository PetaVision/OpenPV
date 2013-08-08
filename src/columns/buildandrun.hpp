/*
 * buildandrun.hpp
 *
 *  Created on: May 27, 2011
 *      Author: peteschultz
 */

#ifndef BUILDANDRUN_HPP_
#define BUILDANDRUN_HPP_

#include <time.h>
#include <string>
#include <iostream>

#include "../include/pv_common.h"

#include "../columns/HyPerCol.hpp"

#include "../layers/HyPerLayer.hpp"
#include "../layers/PVLayer.h"
#include "../layers/ANNLayer.hpp"
#include "../layers/GenerativeLayer.hpp"
#include "../layers/IncrementLayer.hpp"
#include "../layers/LeakyIntegrator.hpp"
#include "../layers/LogLatWTAGenLayer.hpp"
#include "../layers/PursuitLayer.hpp"
#include "../layers/CliqueLayer.hpp"
#include "../layers/PoolingANNLayer.hpp"
#include "../layers/PtwiseProductLayer.hpp"
#include "../layers/TrainingLayer.hpp"
#include "../layers/GapLayer.hpp"
#include "../layers/MaxPooling.hpp"
#include "../layers/TextStream.hpp"
#ifdef PV_USE_SNDFILE
#include "../layers/SoundStream.hpp"
#endif
#include "../layers/Image.hpp"
#include "../layers/CreateMovies.hpp"
#include "../layers/ImageCreator.hpp"
#include "../layers/Movie.hpp"
#include "../layers/Patterns.hpp"
#include "../layers/LabelLayer.hpp"
#include "../layers/LIF.hpp"
#include "../layers/LIFGap.hpp"
#include "../layers/MatchingPursuitLayer.hpp"
#include "../layers/Retina.hpp"
#include "../layers/SigmoidLayer.hpp"
#include "../layers/RescaleLayer.hpp"
#include "../layers/ShuffleLayer.hpp"
#include "../layers/ANNSquaredLayer.hpp"
#include "../layers/ANNWhitenedLayer.hpp"
#include "../layers/ANNDivInh.hpp"
#include "../layers/BIDSLayer.hpp"
#include "../layers/BIDSCloneLayer.hpp"
#include "../layers/BIDSMovieCloneMap.hpp"
#include "../layers/BIDSSensorLayer.hpp"
#include "../layers/LCALIFLayer.hpp"
#include "../layers/LCALayer.hpp"
#include "../layers/HyPerLCALayer.hpp"
#include "../layers/ANNErrorLayer.hpp"
#include "../layers/ANNLabelLayer.hpp"
#include "../layers/ANNTriggerUpdateOnNewImageLayer.hpp"
#include "../connections/HyPerConn.hpp"
#include "../connections/BIDSConn.hpp"
#include "../connections/KernelConn.hpp"
#include "../connections/CliqueConn.hpp"
#include "../connections/CloneKernelConn.hpp"
#include "../connections/GapConn.hpp"
#include "../connections/GenerativeConn.hpp"
#include "../connections/PoolingGenConn.hpp"
#include "../connections/IdentConn.hpp"
#include "../connections/LCAConn.hpp"
#include "../connections/LCALIFLateralKernelConn.hpp"
#include "../connections/NoSelfKernelConn.hpp"
#include "../connections/SiblingConn.hpp"
#include "../connections/OjaKernelConn.hpp"
#include "../connections/ReciprocalConn.hpp"
#include "../connections/TransposeConn.hpp"
#include "../connections/FeedbackConn.hpp"
#include "../connections/LCALIFLateralConn.hpp"
#include "../connections/OjaSTDPConn.hpp"
#include "../connections/InhibSTDPConn.hpp"
#include "../connections/STDP3Conn.hpp"
#include "../connections/STDPConn.hpp"

#include "../weightinit/InitWeights.hpp"
#include "../weightinit/InitGauss2DWeights.hpp"
#include "../weightinit/InitCocircWeights.hpp"
#include "../weightinit/InitSmartWeights.hpp"
#include "../weightinit/InitUniformRandomWeights.hpp"
#include "../weightinit/InitGaussianRandomWeights.hpp"
#include "../weightinit/InitGaborWeights.hpp"
#include "../weightinit/InitDistributedWeights.hpp"
#include "../weightinit/InitBIDSLateral.hpp"
#include "../weightinit/InitPoolWeights.hpp"
#include "../weightinit/InitRuleWeights.hpp"
#include "../weightinit/InitSubUnitWeights.hpp"
#include "../weightinit/InitOneToOneWeights.hpp"
#include "../weightinit/InitIdentWeights.hpp"
#include "../weightinit/InitUniformWeights.hpp"
#include "../weightinit/InitByArborWeights.hpp"
#include "../weightinit/InitSpreadOverArborsWeights.hpp"
#include "../weightinit/Init3DGaussWeights.hpp"
#include "../weightinit/InitWindowed3DGaussWeights.hpp"
#include "../weightinit/InitMTWeights.hpp"

#include "../io/BaseConnectionProbe.hpp"
#include "../io/ReciprocalEnergyProbe.hpp"
#include "../io/KernelProbe.hpp"
#include "../io/LCALIFLateralProbe.hpp"
#include "../io/OjaConnProbe.hpp"
#include "../io/OjaKernelSpikeRateProbe.hpp"
#include "../io/PatchProbe.hpp"
#include "../io/TextStreamProbe.hpp"
#include "../io/LCAProbe.hpp"
#include "../io/LayerProbe.hpp"
#include "../io/PointProbe.hpp"
#include "../io/PointLCALIFProbe.hpp"
#include "../io/PointLIFProbe.hpp"
#include "../io/StatsProbe.hpp"
#include "../io/L2NormProbe.hpp"
#include "../io/SparsityTermProbe.hpp"
#include "../io/GenColProbe.hpp"
#include "../io/LogLatWTAProbe.hpp"

using namespace PV;

int buildandrun(int argc, char * argv[],
                int (*custominit)(HyPerCol *, int, char **) = NULL,
                int (*customexit)(HyPerCol *, int, char **) = NULL,
                void * (*customgroups)(const char *, const char *, HyPerCol *) = NULL);
int buildandrun1paramset(int argc, char * argv[],
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         void * (*customgroups)(const char *, const char *, HyPerCol *),
                         PVParams * params);
HyPerCol * build(int argc, char * argv[], void * (*customgroups)(const char *, const char *, HyPerCol *) = NULL, PVParams * params = NULL);

HyPerCol * addHyPerColToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
HyPerLayer * addLayerToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
TrainingLayer * addTrainingLayer(const char * name, HyPerCol *hc);
GapLayer * addGapLayer(const char * name, HyPerCol * hc);
TextStream * addTextStream(const char * name, HyPerCol *hc);
#ifdef PV_USE_SNDFILE
SoundStream * addSoundStream(const char * name, HyPerCol *hc);
#endif
Image * addImage(const char * name, HyPerCol *hc);
Movie * addMovie(const char * name, HyPerCol *hc);
Patterns * addPatterns(const char * name, HyPerCol *hc);
LabelLayer * addLabelLayer(const char * name, HyPerCol * hc);
ANNTriggerUpdateOnNewImageLayer * addANNTriggerUpdateOnNewImageLayer(const char * name, HyPerCol * hc);
SigmoidLayer * addSigmoidLayer(const char * name, HyPerCol * hc);
RescaleLayer * addRescaleLayer(const char * name, HyPerCol * hc);
ShuffleLayer * addShuffleLayer(const char * name, HyPerCol * hc);
BIDSCloneLayer * addBIDSCloneLayer(const char * name, HyPerCol * hc);
InitWeights *createInitWeightsObject(const char * name, HyPerCol * hc);
InitWeights * getDefaultInitWeightsMethod(const char * keyword);
HyPerConn * addConnToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
PoolingGenConn * addPoolingGenConn(const char * name, HyPerCol * hc, const char * pre_layer_name, const char * post_layer_name, const char * filename, InitWeights *weightInit);
ColProbe * addColProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
void insertColProbe(ColProbe * colProbe, HyPerCol * hc, const char * classkeyword);
BaseConnectionProbe * addBaseConnectionProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
const char * getStringValueFromParameterGroup(const char * groupName, PVParams * params, const char * parameterString, bool warnIfAbsent);
#ifdef OBSOLETE // Marked obsolete July 3, 2013.  No longer pass HyPerLayers to the connections' constructors, but names
                // of the layers.  Accordingly, use HyPerConn::getPreAndPostLayerNames() instead.
int getPreAndPostLayers(const char * name, HyPerCol * hc, HyPerLayer ** preLayerPtr, HyPerLayer **postLayer);
#endif // OBSOLETE
HyPerLayer * getLayerFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbsent=true);
HyPerConn * getConnFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbesnt=true);
LayerProbe * addLayerProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
int getLayerFunctionProbeParameters(const char * name, const char * keyword, HyPerCol * hc, HyPerLayer ** targetLayer, char ** message, const char ** filename);
int decodeChannel(int channel, ChannelType * channelType);
int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc);

#endif /* BUILDANDRUN_HPP_ */
