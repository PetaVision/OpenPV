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
#include "../layers/ANNLayer.hpp"
#include "../layers/AccumulateLayer.hpp"
#include "../layers/GenerativeLayer.hpp"
#include "../layers/IncrementLayer.hpp"
#include "../layers/LeakyIntegrator.hpp"
#include "../layers/LogLatWTAGenLayer.hpp"
#include "../layers/PursuitLayer.hpp"
#include "../layers/CliqueLayer.hpp"
#include "../layers/MatchingPursuitResidual.hpp"
#include "../layers/PoolingANNLayer.hpp"
#include "../layers/PtwiseProductLayer.hpp"
#include "../layers/TrainingLayer.hpp"
#include "../layers/CloneVLayer.hpp"
#include "../layers/BinningLayer.hpp"
#include "../layers/WTALayer.hpp"
#include "../layers/GapLayer.hpp"
#include "../layers/MaxPooling.hpp"
#include "../layers/TextStream.hpp"
#ifdef PV_USE_SNDFILE
#include "../layers/SoundStream.hpp"
#include "../layers/NewCochlear.h"
#endif
#include "../layers/Image.hpp"
#include "../layers/CreateMovies.hpp"
#include "../layers/Movie.hpp"
#include "../layers/Patterns.hpp"
#include "../layers/LabelLayer.hpp"
#include "../layers/LIF.hpp"
#include "../layers/LIFGap.hpp"
#include "../layers/MatchingPursuitLayer.hpp"
#include "../layers/Retina.hpp"
#include "../layers/SigmoidLayer.hpp"
#include "../layers/MLPSigmoidLayer.hpp"
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
#include "../layers/HyPerLCALayer.hpp"
#include "../layers/ANNErrorLayer.hpp"
#include "../layers/ANNNormalizedErrorLayer.hpp"
#include "../layers/MLPErrorLayer.hpp"
#include "../layers/MLPForwardLayer.hpp"
#include "../layers/MLPOutputLayer.hpp"
#include "../layers/LabelErrorLayer.hpp"
#include "../layers/ANNLabelLayer.hpp"
#ifdef OBSOLETE // Marked obsolete April 23, 2014.
// Use ANNLayer with triggerFlag set to true and triggerLayerName for the triggering layer
#include "../layers/ANNTriggerUpdateOnNewImageLayer.hpp"
#endif // OBSOLETE
#include "../layers/ConstantLayer.hpp"
#include "../connections/HyPerConn.hpp"
#include "../connections/BIDSConn.hpp"
#include "../connections/CloneConn.hpp"
#include "../connections/PlasticCloneConn.hpp"
#include "../connections/KernelConn.hpp"
#include "../connections/ImprintConn.hpp"
#include "../connections/MapReduceKernelConn.hpp"
#include "../connections/CliqueConn.hpp"
#include "../connections/CloneKernelConn.hpp"
#include "../connections/GapConn.hpp"
#include "../connections/GenerativeConn.hpp"
#include "../connections/PoolingGenConn.hpp"
#include "../connections/IdentConn.hpp"
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
#include "../weightinit/InitOneToOneWeightsWithDelays.hpp"
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
#include "../io/LayerProbe.hpp"
#include "../io/PointProbe.hpp"
#include "../io/PointLCALIFProbe.hpp"
#include "../io/PointLIFProbe.hpp"
#include "../io/StatsProbe.hpp"
#include "../io/SparsityLayerProbe.hpp"
#include "../io/L2NormProbe.hpp"
#include "../io/SparsityTermProbe.hpp"
#include "../io/LogLatWTAProbe.hpp"
#include "../io/RequireAllZeroActivityProbe.hpp"
#include "../io/GenColProbe.hpp"

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
HyPerConn * addConnToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
ColProbe * addColProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
void insertColProbe(ColProbe * colProbe, HyPerCol * hc);
BaseConnectionProbe * addBaseConnectionProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
const char * getStringValueFromParameterGroup(const char * groupName, PVParams * params, const char * parameterString, bool warnIfAbsent);
HyPerLayer * getLayerFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbsent=true);
HyPerConn * getConnFromParameterGroup(const char * groupName, HyPerCol * hc, const char * parameterStringName, bool warnIfAbesnt=true);
LayerProbe * addLayerProbeToColumn(const char * classkeyword, const char * name, HyPerCol * hc);
int getLayerFunctionProbeParameters(const char * name, const char * keyword, HyPerCol * hc, HyPerLayer ** targetLayer, char ** message, const char ** filename);
int decodeChannel(int channel, ChannelType * channelType);
int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc);

#endif /* BUILDANDRUN_HPP_ */
