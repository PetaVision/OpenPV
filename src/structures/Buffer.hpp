#ifndef __BUFFER_HPP__
#define __BUFFER_HPP__

#include <vector>

namespace PV {

template <class T>
class Buffer {
  public:
   enum Anchor { CENTER, NORTH, NORTHEAST, EAST, SOUTHEAST, SOUTH, SOUTHWEST, WEST, NORTHWEST };

   Buffer(int width, int height, int features);
   Buffer();
   Buffer(const std::vector<T> &data, int width, int height, int features);
   Buffer(const T *data, int width, int height, int features);
   T const at(int x, int y, int feature) const;
   T const at(int k) const;
   void set(int x, int y, int feature, T value);
   void set(int k, T value);
   void set(const std::vector<T> &vector, int width, int height, int features);
   void set(const T *data, int width, int height, int features);
   void set(Buffer<T> other);
   void resize(int width, int height, int features);
   void crop(int newWidth, int newHeight, enum Anchor anchor);
   void grow(int newWidth, int newHeight, enum Anchor anchor);
   void translate(int offsetX, int offsetY);
   void flip(bool xFlip, bool yFlip);
   std::vector<T> asVector() const { return mData; }
   int getHeight() const { return mHeight; }
   int getWidth() const { return mWidth; }
   int getFeatures() const { return mFeatures; }
   int getTotalElements() const { return mHeight * mWidth * mFeatures; }

  protected:
   static int getAnchorX(enum Anchor anchor, int smallerWidth, int biggerWidth);
   static int getAnchorY(enum Anchor anchor, int smallerHeight, int biggerHeight);
   inline int index(int x, int y, int f) const { return f + (x + y * mWidth) * mFeatures; }

   std::vector<T> mData;
   int mWidth    = 0;
   int mHeight   = 0;
   int mFeatures = 0;
}; // end class Buffer

} // end namespace PV

#include "Buffer.tpp" // template implementations file

#endif
