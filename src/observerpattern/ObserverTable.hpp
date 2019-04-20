/*
 * ObserverTable.hpp
 *
 *  Created on: Nov 20, 2017
 *      Author: pschultz
 */

#ifndef OBSERVERTABLECOMPONENT_HPP_
#define OBSERVERTABLECOMPONENT_HPP_

#include "observerpattern/Observer.hpp"
#include <map>
#include <vector>

namespace PV {

/**
 * An Observer subclass containing a table of Observer objects. map as its main data member
 * Objects can be added with the addObserver method.
 * The table can be searched by name using the lookupByName or lookupByType templates.
 * The lookup tables also have recursive forms: if an object is not found in the table
 * itself, but the table has an ObserverTable as one of its Observer objects,
 * that ObserverTable is then searched, and so on.
 *
 * The motivation is that components within a layer or connection may need to find other
 * objects in the HyPerCol. The HyPerCol creates an ObserverTable containing its
 * rest of its hierarchy, and then adds it to the hierarchy. The layers and connections
 * then have access to the ObserverTable and its contents during the
 * CommunicateInitInfo stage.
 */
class ObserverTable : public Observer {
  public:
   ObserverTable(char const *description);

   virtual ~ObserverTable();

   /**
    * Adds an Observer object to the table. Internally, the table is stored in
    * two forms, as a vector and as a map with strings as the search index.
    * When searching for an object by name, the map is used for efficient lookup.
    * When iterating in a for loop, the vector is used, to preserve the order
    * in which the objects are called.
    */
   void addObject(std::string const &name, Observer *entry);

   void copyTable(ObserverTable const *origTable);

   template <typename T>
   T *findObject(std::string const &name) const;

   template <typename T>
   T *findObject(char const *name) const;

   template <typename T>
   std::vector<T *> findObjects(std::string const &name) const;

   template <typename T>
   std::vector<T *> findObjects(char const *name) const;

   // To iterate over ObserverTable:
   typedef std::vector<Observer *>::iterator iterator;
   typedef std::vector<Observer *>::const_iterator const_iterator;
   typedef std::vector<Observer *>::reverse_iterator reverse_iterator;
   typedef std::vector<Observer *>::const_reverse_iterator const_reverse_iterator;

   iterator begin() { return mTableAsVector.begin(); }
   const_iterator begin() const { return mTableAsVector.begin(); }
   const_iterator cbegin() const { return mTableAsVector.cbegin(); }
   iterator end() { return mTableAsVector.end(); }
   const_iterator end() const { return mTableAsVector.end(); }
   const_iterator cend() const { return mTableAsVector.cend(); }

   /**
    * Empties the table of components. It does not delete or free the individual components
    * in the table; it only drops the pointers to them.
    */
   void clear();

  protected:
   ObserverTable();

   void initialize(char const *description);

  protected:
   std::vector<Observer *> mTableAsVector;
   std::multimap<std::string, Observer *> mTableAsMultimap;

}; // class ObserverTable

} // namespace PV

// Template method implementations
#include "ObserverTable.tpp"

#endif // OBSERVERTABLECOMPONENT_HPP_
