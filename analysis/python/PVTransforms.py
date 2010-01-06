def kxPos(k, nx, ny, nf):
   ''' * Return the position kx for the given k index
       * @k the k index (can be either global or local depending on if nx,ny are global or local)
       * @nx the number of neurons in the x direction
       * @ny the number of neurons in the y direction'''
   return k % nx
#end of kxPos


def kyPos(k, nx, ny, nf):
   ''' * Return the position ky for the given k index
       * @k the k index (can be either global or local depending on if nx,ny are global or local)
       * @nx the number of neurons in the x direction
       * @ny the number of neurons in the y direction'''             
   return k / ny
#end of kyPos
