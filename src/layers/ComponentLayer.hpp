#pragma once

#include "HyPerLayer.hpp"
#include <layers/components/Component.hpp>

namespace PV
{
	class InputComponent;
	class StateComponent;
	class OutputComponent;
	
	class ComponentLayer : public HyPerLayer
	{
		public:
			ComponentLayer(const char * name, HyPerCol * hc);
			~ComponentLayer();

			virtual int resetGSynBuffers(double timef, double dt);
			virtual int recvAllSynapticInput();
			virtual bool activityIsSpiking() { return false; }
			
			int calcActivityIndex(int k);
			int calcBatchOffset();
		
		protected:
			int initialize(const char * name, HyPerCol * hc);
		
			virtual int allocateDataStructures();
			virtual int allocateGSyn();
			virtual int allocateV();
			virtual int allocateActivity();
			virtual int setActivity();
			virtual int initializeV();
			virtual int initializeActivity();
			virtual int callUpdateState(double timed, double dt);
			virtual int readActivityFromCheckpoint(const char * cpDir, double * timeptr);
			virtual int readVFromCheckpoint(const char * cpDir, double * timeptr);
			virtual int ioParamsFillGroup(enum ParamsIOFlag ioFlag);
			
			virtual void ioParam_callDerivative(enum ParamsIOFlag ioFlag);
			virtual void ioParam_inputComponent(enum ParamsIOFlag ioFlag);
			virtual void ioParam_stateComponent(enum ParamsIOFlag ioFlag);
			virtual void ioParam_outputComponent(enum ParamsIOFlag ioFlag);
			
		private:
			int initialize_base();
			bool mCallDerivative;
			InputComponent *mInputComponent = nullptr;
			StateComponent *mStateComponent = nullptr;
			OutputComponent *mOutputComponent = nullptr;
			char *mInputType = nullptr;
			char *mStateType = nullptr;
			char *mOutputType = nullptr;
	};
	
	BaseObject * createComponentLayer(char const * name, HyPerCol * hc);
}
