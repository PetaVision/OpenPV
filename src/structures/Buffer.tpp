#include "utils/PVLog.hpp"
//#include "utils/conversions.hpp"

#include <cmath>
#include <cstring>

namespace PV {

template <class T>
Buffer<T>::Buffer(int width, int height, int features) {
   resize(width, height, features);
}

template <class T>
Buffer<T>::Buffer() {
   resize(1, 1, 1);
}

template <class T>
Buffer<T>::Buffer(const std::vector<T> &data, int width, int height, int features) {
   set(data, width, height, features);
}

// This constructor is for backwards compatibility with raw float buffers.
// It lacks bounds checking and should be removed when layers used Buffers
// instead of malloc'd floats.
template <class T>
Buffer<T>::Buffer(const T *data, int width, int height, int features) {
   set(data, width, height, features);
}

template <class T>
T const Buffer<T>::at(int x, int y, int feature) const {
   return at(index(x, y, feature));
}

template <class T>
T const Buffer<T>::at(int k) const {
   return mData.at(k);
}

template <class T>
void Buffer<T>::set(int x, int y, int feature, T value) {
   set(index(x, y, feature), value);
}

template <class T>
void Buffer<T>::set(int k, T value) {
   mData.at(k) = value;
}

template <class T>
void Buffer<T>::set(const std::vector<T> &vector, int width, int height, int features) {
   FatalIf(
         (int)vector.size() != width * height * features,
         "Invalid vector size: Expected %d elements, vector contained %d elements.\n",
         width * height * features,
         vector.size());
   mData     = vector;
   mWidth    = width;
   mHeight   = height;
   mFeatures = features;
}

template <class T>
void Buffer<T>::set(const T *data, int width, int height, int features) {
   std::vector<T> tempVector(width * height * features);
   for (size_t i = 0; i < tempVector.size(); ++i) {
      tempVector.at(i) = data[i];
   }
   set(tempVector, width, height, features);
}

template <class T>
void Buffer<T>::set(Buffer<T> other) {
   set(other.asVector(), other.getWidth(), other.getHeight(), other.getFeatures());
}

template <class T>
void Buffer<T>::insert(Buffer<T> const &insertion, int xStart, int yStart) {
   int xEnd = xStart + insertion.getWidth();
   FatalIf(xStart < 0 or xStart >= getWidth(), "Buffer::insert() has bad xStart = %d (Width = %d)\n", xStart, getWidth());
   FatalIf(xEnd <= 0 or xEnd > getWidth(),
         "Buffer::insert() has bad x interval = (%d,%d) (Width = %d; insertion width = %d)\n",
         xStart, xEnd, getWidth(), insertion.getWidth());

   int yEnd = yStart + insertion.getHeight();
   FatalIf(yStart < 0 or yStart >= getHeight(), "Buffer::insert() has bad yStart = %d (Height = %d)\n", yStart, getHeight());
   FatalIf(yEnd <= 0 or yEnd > getHeight(),
         "Buffer::insert() has bad y interval = (%d,%d) (Height = %d; insertion height = %d)\n",
         yStart, yEnd, getHeight(), insertion.getHeight());

   FatalIf(getFeatures() != insertion.getFeatures(),
         "Buffer::insert() has incompatible number of features: %d versus %d\n",
         getFeatures(), insertion.getFeatures());

   for (int y = 0; y < insertion.getHeight(); ++y) {
      for (int x = 0; x < insertion.getWidth(); ++x) {
         for (int f = 0; f < insertion.getFeatures(); ++f) {
            T value = insertion.at(x, y, f);
            set(x + xStart, y + yStart, f, value);
         }
      }
   }
}

template <class T>
Buffer<T> Buffer<T>::extract(int xStart, int yStart, int width, int height) const {
   FatalIf(xStart < 0 or xStart >= getWidth(), "Buffer::extract() has bad xStart %d (Width = %d)\n", xStart, getWidth());
   FatalIf(xStart + width < 0 or xStart + width > getWidth(),
         "Buffer::extract() has bad width %d (xStart = %d; total width = %d)\n", width, xStart, getWidth());
   FatalIf(yStart < 0 or yStart >= getHeight(), "Buffer::extract() has bad yStart %d (Height = %d)\n", yStart, getHeight());
   FatalIf(yStart + height < 0 or yStart + height > getHeight(),
         "Buffer::extract() has bad height %d (yStart = %d; total height = %d)\n", height, yStart, getHeight());

   Buffer<T> result(width, height, getFeatures());
   for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
         for (int f = 0; f < getFeatures(); ++f) {
            T value = at(x + xStart, y + yStart, f);
            result.set(x, y, f, value);
         }
      }
   }
   return result;
}

// Resizing a Buffer will clear its contents. Use rescale, crop, or grow to preserve values.
template <class T>
void Buffer<T>::resize(int width, int height, int features) {
   mData.clear();
   mData.resize(height * width * features);
   mWidth    = width;
   mHeight   = height;
   mFeatures = features;
}

// Grows a buffer
template <class T>
void Buffer<T>::grow(int newWidth, int newHeight, enum Anchor anchor) {
   if (newWidth <= getWidth() && newHeight <= getHeight()) {
      return;
   }
   newWidth    = std::max(newWidth, getWidth());
   newHeight   = std::max(newHeight, getHeight());
   int offsetX = getAnchorX(anchor, getWidth(), newWidth);
   int offsetY = getAnchorY(anchor, getHeight(), newHeight);
   Buffer bigger(newWidth, newHeight, getFeatures());

   for (int y = 0; y < getHeight(); ++y) {
      for (int x = 0; x < getWidth(); ++x) {
         for (int f = 0; f < getFeatures(); ++f) {
            int destX = x + offsetX;
            int destY = y + offsetY;
            if (destX < 0 || destX >= newWidth)
               continue;
            if (destY < 0 || destY >= newHeight)
               continue;
            bigger.set(destX, destY, f, at(x, y, f));
         }
      }
   }
   set(bigger.asVector(), newWidth, newHeight, getFeatures());
}

// Crops a buffer down to the specified size
template <class T>
void Buffer<T>::crop(int newWidth, int newHeight, enum Anchor anchor) {
   if (newWidth >= getWidth() && newHeight >= getHeight()) {
      return;
   }
   int offsetX = getAnchorX(anchor, newWidth, getWidth());
   int offsetY = getAnchorY(anchor, newHeight, getHeight());
   Buffer cropped(newWidth, newHeight, getFeatures());

   for (int destY = 0; destY < newHeight; ++destY) {
      for (int destX = 0; destX < newWidth; ++destX) {
         for (int f = 0; f < getFeatures(); ++f) {
            int sourceX = destX + offsetX;
            int sourceY = destY + offsetY;
            if (sourceX < 0 || sourceX >= getWidth())
               continue;
            if (sourceY < 0 || sourceY >= getHeight())
               continue;
            cropped.set(destX, destY, f, at(sourceX, sourceY, f));
         }
      }
   }
   set(cropped.asVector(), newWidth, newHeight, getFeatures());
}

template <class T>
void Buffer<T>::flip(bool xFlip, bool yFlip) {
   if (!xFlip && !yFlip) {
      return;
   }
   Buffer result(getWidth(), getHeight(), getFeatures());
   for (int y = 0; y < getHeight(); ++y) {
      for (int x = 0; x < getWidth(); ++x) {
         for (int f = 0; f < getFeatures(); ++f) {
            int destX = xFlip ? getWidth() - 1 - x : x;
            int destY = yFlip ? getHeight() - 1 - y : y;
            result.set(destX, destY, f, at(x, y, f));
         }
      }
   }
   set(result.asVector(), getWidth(), getHeight(), getFeatures());
}

// Shift a buffer, clipping any values that land out of bounds
template <class T>
void Buffer<T>::translate(int xShift, int yShift) {
   if (xShift == 0 && yShift == 0) {
      return;
   }
   Buffer result(getWidth(), getHeight(), getFeatures());
   for (int y = 0; y < getHeight(); ++y) {
      for (int x = 0; x < getWidth(); ++x) {
         for (int f = 0; f < getFeatures(); ++f) {
            int destX = x + xShift;
            int destY = y + yShift;
            if (destX < 0 || destX >= getWidth())
               continue;
            if (destY < 0 || destY >= getHeight())
               continue;
            result.set(destX, destY, f, at(x, y, f));
         }
      }
   }
   set(result.asVector(), getWidth(), getHeight(), getFeatures());
}

template <class T>
int Buffer<T>::getAnchorX(enum Anchor anchor, int smallerWidth, int biggerWidth) {
   int resultX;
   switch (anchor) {
      case NORTHWEST:
      case WEST:
      case SOUTHWEST: resultX = 0; break;
      case NORTH:
      case CENTER:
      case SOUTH: resultX = biggerWidth / 2 - smallerWidth / 2; break;
      case NORTHEAST:
      case EAST:
      case SOUTHEAST: resultX = biggerWidth - smallerWidth; break;
      default: resultX        = 0; break;
   }
   return resultX;
}

template <class T>
int Buffer<T>::getAnchorY(enum Anchor anchor, int smallerHeight, int biggerHeight) {
   int resultY;
   switch (anchor) {
      case NORTHWEST:
      case NORTH:
      case NORTHEAST: resultY = 0; break;
      case WEST:
      case CENTER:
      case EAST: resultY = biggerHeight / 2 - smallerHeight / 2; break;
      case SOUTHWEST:
      case SOUTH:
      case SOUTHEAST: resultY = biggerHeight - smallerHeight; break;
      default: resultY        = 0; break;
   }
   return resultY;
}

} // end namespace PV
