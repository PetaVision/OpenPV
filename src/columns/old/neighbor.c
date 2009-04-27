#include <columns/PVHyperCol.h>

#include <assert.h>
#include <stdio.h>


static int hasWesternNeighbor(PVHyperCol* hc, int comm_id)
{
    return comm_id % hc->n_cols;
}


static int hasEasternNeighbor(PVHyperCol* hc, int comm_id)
{
    return (comm_id + 1) % hc->n_cols;
}


static int hasNorthernNeighbor(PVHyperCol* hc, int comm_id)
{
    return ((comm_id + hc->n_cols) > (hc->comm_size - 1)) ? 0 : 1;
}


static int hasSouthernNeighbor(PVHyperCol* hc, int comm_id)
{
    return ((comm_id - hc->n_cols) < 0) ? 0 : 1;
}


int pv_neighbor_init(PVHyperCol* hc)
  {
    int i, n;
    int n_neighbors = 0;

    /* initialize neighbors lists */

    hc->n_neighbors = pv_number_neighbors(hc, hc->comm_id);

    for (i = 0; i < NUM_NEIGHBORS; i++)
      {
        hc->neighbors[i] = 0;
        n = neighbor_index(hc, hc->comm_id, i);
        if (n >= 0)
          {
            hc->neighbors[n_neighbors++] = n;
            if (DEBUG)
              printf("[%d] init_neighbors: neighbor[%d] of %d is %d, i = %d\n",
                  hc->comm_id, n_neighbors-1, hc->n_neighbors, n, i);
          }
      }
    assert(hc->n_neighbors == n_neighbors);
    
    return 0;
  }


int pv_number_neighbors(PVHyperCol* hc, int comm_id)
{
    int n = 0;

    int hasWest  = hasWesternNeighbor(hc, comm_id);
    int hasEast  = hasEasternNeighbor(hc, comm_id);
    int hasNorth = hasNorthernNeighbor(hc, comm_id);
    int hasSouth = hasSouthernNeighbor(hc, comm_id);
  
    if (hasNorth > 0) n += 1;
    if (hasSouth > 0) n += 1;

    if (hasWest > 0) {
        n += 1;
        if (hasNorth > 0) n += 1;
        if (hasSouth > 0) n += 1;
    }

    if (hasEast > 0) {
        n += 1;
        if (hasNorth > 0) n += 1;
        if (hasSouth > 0) n += 1;
    }

    return n;
}

/**
 * Returns the comm_id of the northwestern HyperColumn
 */
int pv_northwest(PVHyperCol* hc, int comm_id)
{
    if (hasNorthernNeighbor(hc, comm_id) == 0) return -1;
    return pv_west(hc, comm_id + hc->n_cols);
}


/**
 * Returns the comm_id of the northern HyperColumn
 */
int pv_north(PVHyperCol* hc, int comm_id)
{
    if (hasNorthernNeighbor(hc, comm_id) == 0) return -1;
    return (comm_id + hc->n_cols);
}


/**
 * Returns the comm_id of the northeastern HyperColumn
 */
int pv_northeast(PVHyperCol* hc, int comm_id)
{
    if (hasNorthernNeighbor(hc, comm_id) == 0) return -1;
    return pv_east(hc, comm_id + hc->n_cols);
}


/**
 * Returns the comm_id of the western HyperColumn
 */
int pv_west(PVHyperCol* hc, int comm_id)
{
    if (hasWesternNeighbor(hc, comm_id) == 0) return -1;
    return (pv_hypercol_row(hc, comm_id)*hc->n_cols + ((comm_id - 1) % hc->n_cols));
}


/**
 * Returns the comm_id of the eastern HyperColumn
 */
int pv_east(PVHyperCol* hc, int comm_id)
{
    if (hasEasternNeighbor(hc, comm_id) == 0) return -1;
    return (pv_hypercol_row(hc, comm_id)*hc->n_cols + ((comm_id + 1) % hc->n_cols));
}


/**
 * Returns the comm_id of the southwestern HyperColumn
 */
int pv_southwest(PVHyperCol* hc, int comm_id)
{
    if (hasSouthernNeighbor(hc, comm_id) == 0) return -1;
    return pv_west(hc, comm_id - hc->n_cols);
}


/**
 * Returns the comm_id of the southern HyperColumn
 */
int pv_south(PVHyperCol* hc, int comm_id)
{
    if (hasSouthernNeighbor(hc, comm_id) == 0) return -1;
    return (comm_id - hc->n_cols);
}


/**
 * Returns the comm_id of the southeastern HyperColumn
 */
int pv_southeast(PVHyperCol* hc, int comm_id)
{
    if (hasSouthernNeighbor(hc, comm_id) == 0) return -1;
    return pv_east(hc, comm_id - hc->n_cols);
}


/**
 * Returns the sender rank for the given connection index
 */
int neighbor_index(PVHyperCol* hc, int comm_id, int index)
{
    switch (index) {
      case 0: /* northwest */
        return pv_northwest(hc, comm_id);
      case 1: /* north */
        return pv_north(hc, comm_id);
      case 2: /* northeast */
        return pv_northeast(hc, comm_id);
      case 3: /* west */
        return pv_west(hc, comm_id);
      case 4: /* east */
        return pv_east(hc, comm_id);
      case 5: /* southwest */
        return pv_southwest(hc, comm_id);
      case 6: /* south */
        return pv_south(hc, comm_id);
      case 7: /* southeast */
        return pv_southeast(hc, comm_id);
      default:
        fprintf(stderr, "ERROR:neighbor_index: bad index\n");
    }   
    return -1;
}

