#pragma once

#include "OutputComponent.hpp"

#include <limits>

namespace PV
{
	class OutputComponent;
	
	class ANNOutputComponent : public OutputComponent
	{
		public:
			~ANNOutputComponent();
			
			virtual void transform();
			virtual void derivative();
			virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag);
			virtual void initialize();
			
		protected:
			pvdata_t AMax;
			pvdata_t AMin;
			pvdata_t VThresh;
			pvdata_t AShift;
			pvdata_t VWidth;
			int numVertices = 0;
			float slopeNegInf = 1.0f;
			float slopePosInf = 1.0f;
			float *slopes;
			pvpotentialdata_t *verticesV;
			pvadata_t *verticesA;
			
			void setSlopes();
			void setVertices();
			virtual void ioParam_VThresh(enum ParamsIOFlag ioFlag);
			virtual void ioParam_AMin(enum ParamsIOFlag ioFlag);
			virtual void ioParam_AMax(enum ParamsIOFlag ioFlag);
			virtual void ioParam_AShift(enum ParamsIOFlag ioFlag);
			virtual void ioParam_VWidth(enum ParamsIOFlag ioFlag);

	};
	
	BaseObject * createANNOutputComponent(char const * name, HyPerCol * hc);
}

