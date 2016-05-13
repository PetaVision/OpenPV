#pragma once

#include "Component.hpp"
#include <layers/ComponentLayer.hpp>
#include <columns/BaseObject.hpp>

namespace PV
{
	class ComponentLayer;
	
	template<class BufferType>
	class Component : public BaseObject
	{
		public:
			~Component<BufferType>()
			{
				//The layer still owns mTransformBuffer, since it's just a pointer
				//to the correct clayer buffer. The layer will free it.
				//We free mDerivativeBuffer here.
				//TODO: Give ownership of buffers to components, so they allocate
				//and free both of their buffers.
				if(mDerivativeBuffer != nullptr) delete [] mDerivativeBuffer;
			}
			virtual BufferType* getTransformBuffer() { return mTransformBuffer; }
			virtual BufferType* getDerivativeBuffer() { return mDerivativeBuffer; }
			virtual void ioParamsFillGroup(enum ParamsIOFlag ioFlag) { }
			//Not the same as BaseObject::initialize, this is called by the parent
         //ComponentLayer after each buffer's values have been initialized
			virtual void initialize() {}
			virtual void allocateTransformBuffer() = 0;
			virtual void allocateDerivativeBuffer() = 0;
			virtual void transform() = 0;
			virtual void derivative() = 0;
			void setParentLayer(ComponentLayer *parentLayer)
			{
				if(mParentLayer == nullptr) { mParentLayer = parentLayer; }
			}
			//These are kind of hacks because the column doesn't actually build
			//components when it builds everything else, ComponentLayer does it
			void setParentColumn(HyPerCol* parent) { setParent(parent); }
			//void setName(const char* name) { setName(name); }
			
		protected:
			ComponentLayer *mParentLayer = nullptr;
			BufferType* mTransformBuffer = nullptr;
			BufferType* mDerivativeBuffer = nullptr;
	};
}

//This is pretty ugly but it's the only way I could get these to
//share a parent template. Forward declaration doesn't seem to work
//when the base class is a template. Changes or suggestions very welcome.
#include "InputComponent.hpp"
#include "OutputComponent.hpp"
#include "StateComponent.hpp"
