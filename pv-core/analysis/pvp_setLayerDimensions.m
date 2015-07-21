function pvp_setLayerDimensions(layer, pvp_header, pvp_index)

  global N NROWS NCOLS % for the current layer
  global NFEATURES  % for the current layer
  global NO NK % for the current layer
  
  NCOLS = pvp_header(pvp_index.NX);
  NROWS = pvp_header(pvp_index.NY);
  NFEATURES = pvp_header(pvp_index.NF);
  N = NROWS * NCOLS * NFEATURES;
  NO = floor( NFEATURES / NK );
  
