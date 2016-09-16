#ifndef __BUFFER_HPP__
#define __BUFFER_HPP__

#include <vector>

namespace PV {

template <class T> 
class Buffer {
   public:
     
      enum Anchor {
         CENTER,
         NORTH,
         NORTHEAST,
         EAST,
         SOUTHEAST,
         SOUTH,
         SOUTHWEST,
         WEST,
         NORTHWEST
      };

      Buffer(int width, int height, int features); 
      Buffer();
      Buffer(const std::vector<T> &data, int width, int height, int features);
      Buffer(const T* data, int width, int height, int features);
      T at(int x, int y, int feature); 
      void set(int x, int y, int feature, T value);
      void set(const std::vector<T> &vector, int width, int height, int features);
      void set(const T* data, int width, int height, int features);
      void set(Buffer<T> other);
      void resize(int width, int height, int features);
      void crop(int newWidth, int newHeight, enum Anchor anchor);
      void grow(int newWidth, int newHeight, enum Anchor anchor);
      void translate(int offsetX, int offsetY);
      std::vector<T> asVector()    { return mData; }
      const int getHeight()        { return mHeight; }
      const int getWidth()         { return mWidth; }
      const int getFeatures()      { return mFeatures; }
      const int getTotalElements() { return mHeight * mWidth * mFeatures; }
   protected:
      static int getAnchorX(enum Anchor anchor, int smallerWidth, int biggerWidth);
      static int getAnchorY(enum Anchor anchor, int smallerHeight, int biggerHeight);
      inline int index(int x, int y, int f) {
         return f + (x + y * mWidth) * mFeatures;
      }
      
      std::vector<T> mData;
      int mWidth    = 0;
      int mHeight   = 0;
      int mFeatures = 0;      
};
}
#endif
