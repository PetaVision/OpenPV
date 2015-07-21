def kxPos(k, nx, ny, nf):
   ''' * Return the position kx for the given k index
       * @k the k index (can be either global or local depending on if nx,ny are global or local)
       * @nx the number of neurons in the x direction
       * @ny the number of neurons in the y direction'''
   return (k/nf) % nx
#end of kxPos


def kyPos(k, nx, ny, nf):
   ''' * Return the position ky for the given k index
       * @k the k index (can be either global or local depending on if nx,ny are global or local)
       * @nx the number of neurons in the x direction
       * @ny the number of neurons in the y direction'''             
   return k / (nx*nf)
#end of kyPos


def kxBlockedPos(kGlobal, nxGlobal, nyGlobal, nf, nxBlocks, nyBlocks):
   ''' * Return the position kx for the given k index
       * @kGlobal the global k index
       * @nxGlobal the number of neurons in the x direction
       * @nyGlobal the number of neurons in the y direction
       * @nxBlocks the number of blocks in the x direction
       * @nyBlocks the number of blocks in the y direction'''
   nx = nxGlobal / nxBlocks
   ny = nyGlobal / nyBlocks
   block = kGlobal / (nx*ny)
   xBlock = block % nxBlocks
   numBlocks = nxBlocks * nyBlocks
   kInBlock = kGlobal % (nx*ny)
   return nx*xBlock + kxPos(kInBlock, nx, ny, nf)
#end of kxBlockedPos


def kyBlockedPos(kGlobal, nxGlobal, nyGlobal, nf, nxBlocks, nyBlocks):
   ''' * Return the position ky for the given k index
       * @kGlobal the global k index
       * @nxGlobal the number of neurons in the x direction
       * @nyGlobal the number of neurons in the y direction
       * @nxBlocks the number of blocks in the x direction
       * @nyBlocks the number of blocks in the y direction'''
   nx = nxGlobal / nxBlocks
   ny = nyGlobal / nyBlocks
   block = kGlobal / (nx*ny)
   yBlock = block / nxBlocks
   numBlocks = nxBlocks * nyBlocks
   kInBlock = kGlobal % (nx*ny)
   return ny*yBlock + kyPos(kInBlock, nx, ny, nf)
#end of kyBlockedPos
