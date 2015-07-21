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

#include "../weightinit/InitWeights.hpp"
#include "../normalizers/NormalizeBase.hpp"

#ifdef OBSOLETE // Marked obsolete Feb 6, 2015.  buildandrun builds these objects by calling CoreParamGroupHandler, so the include statements are in that class.
#include "../layers/HyPerLayer.hpp"
#include "../layers/ANNLayer.hpp"
#include "../layers/GenerativeLayer.hpp"
#include "../layers/IncrementLayer.hpp"
#include "../layers/LeakyIntegrator.hpp"
#include "../layers/LogLatWTAGenLayer.hpp"
#include "../layers/MatchingPursuitResidual.hpp"
#include "../layers/PoolingANNLayer.hpp"
#include "../layers/PtwiseProductLayer.hpp"
#include "../layers/TrainingLayer.hpp"
#include "../layers/CloneVLayer.hpp"
#include "../layers/BinningLayer.hpp"
#include "../layers/WTALayer.hpp"
#include "../layers/GapLayer.hpp"
#ifdef OBSOLETE // Marked obsolete Mar 16, 2015.  Text-related classes have been moved to auxlib/pvtext
#include "../layers/TextStream.hpp"
#endif // Marked obsolete Mar 16, 2015.  Text-related classes have been moved to auxlib/pvtext
#ifdef OBSOLETE // Marked obsolete Mar 16, 2015.  Sound-related classes have been moved to auxlib/pvsound
#include "../layers/SoundStream.hpp"
#include "../layers/NewCochlear.h"
#endif // Marked obsolete Mar 16, 2015.  Sound-related classes have been moved to auxlib/pvsound
#include "../layers/Image.hpp"
#include "../layers/CreateMovies.hpp"
#include "../layers/ImageFromMemoryBuffer.hpp"
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
#include "../layers/RunningAverageLayer.hpp"
#include "../layers/ShuffleLayer.hpp"
#include "../layers/ANNSquaredLayer.hpp"
#include "../layers/ANNWhitenedLayer.hpp"
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
#include "../layers/KmeansLayer.hpp"
#include "../layers/ConstantLayer.hpp"
#include "../layers/FilenameParsingGroundTruthLayer.hpp"
#include "../connections/BaseConnection.hpp"
#include "../connections/HyPerConn.hpp"
#include "../connections/BIDSConn.hpp"
#include "../connections/CloneConn.hpp"
#include "../connections/PlasticCloneConn.hpp"
#include "../connections/CopyConn.hpp"
#include "../connections/KernelConn.hpp"
#include "../connections/CloneKernelConn.hpp"
#include "../connections/GapConn.hpp"
#include "../connections/IdentConn.hpp"
#include "../connections/TransposeConn.hpp"
#include "../connections/FeedbackConn.hpp"
#include "../connections/LCALIFLateralConn.hpp"
#include "../connections/OjaSTDPConn.hpp"
#include "../connections/PoolingConn.hpp"

#include "../weightinit/InitWeights.hpp"
#include "../weightinit/InitGauss2DWeights.hpp"
#include "../weightinit/InitCocircWeights.hpp"
#include "../weightinit/InitSmartWeights.hpp"
#include "../weightinit/InitUniformRandomWeights.hpp"
#include "../weightinit/InitGaussianRandomWeights.hpp"
#include "../weightinit/InitGaborWeights.hpp"
#include "../weightinit/InitBIDSLateral.hpp"
#include "../weightinit/InitOneToOneWeights.hpp"
#include "../weightinit/InitOneToOneWeightsWithDelays.hpp"
#include "../weightinit/InitIdentWeights.hpp"
#include "../weightinit/InitUniformWeights.hpp"
#include "../weightinit/InitSpreadOverArborsWeights.hpp"
#endif // OBSOLETE // Feb 6, 2015

#ifdef OBSOLETE // Marked obsolete Feb 6, 2015.  buildandrun builds these objects by calling CoreParamGroupHandler, so the include statements are in that class.
#include "../io/BaseConnectionProbe.hpp"
#include "../io/KernelProbe.hpp"
#include "../io/TextStreamProbe.hpp"
#include "../io/LayerProbe.hpp"
#include "../io/PointProbe.hpp"
#include "../io/PointLIFProbe.hpp"
#include "../io/StatsProbe.hpp"
#include "../io/SparsityLayerProbe.hpp"
#include "../io/L2NormProbe.hpp"
#include "../io/LogLatWTAProbe.hpp"
#include "../io/RequireAllZeroActivityProbe.hpp"
#include "../io/GenColProbe.hpp"
#endif // OBSOLETE // Feb 6, 2015

#include "../io/ParamGroupHandler.hpp"
#include "../io/CoreParamGroupHandler.hpp"

using namespace PV;

// The build, buildandrun1paramset, and buildandrun functions are included for backwards compatibility.  The three versions after them, which use ParamGroupHandler arguments, are preferred.
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

// The build, buildandrun1paramset, and buildandrun functions below are preferred to the versions above.
int buildandrun(int argc, char * argv[],
                int (*custominit)(HyPerCol *, int, char **),
                int (*customexit)(HyPerCol *, int, char **),
                ParamGroupHandler ** groupHandlerList, int numGroupHandlers);
int buildandrun1paramset(int argc, char * argv[],
                         int (*custominit)(HyPerCol *, int, char **),
                         int (*customexit)(HyPerCol *, int, char **),
                         ParamGroupHandler ** groupHandlerList, int numGroupHandlers,
                         PVParams * params);
HyPerCol * build(int argc, char * argv[], ParamGroupHandler ** groupHandlerList, int numGroupHandlers, PVParams * params);
ParamGroupHandler * getGroupHandlerFromList(char const * keyword, CoreParamGroupHandler * coreHandler, ParamGroupHandler ** groupHandlerList, int numGroupHandlers, ParamGroupType * foundGroupType);
BaseConnection * createConnection(CoreParamGroupHandler * coreGroupHandler, ParamGroupHandler ** customHandlerList, int numGroupHandlers, char const * keyword, char const * groupname, HyPerCol * hc);

int checknewobject(void * object, const char * kw, const char * name, HyPerCol * hc);

#endif /* BUILDANDRUN_HPP_ */
