#
#     grabcut.py
#     Vadas Gintautas 2011
#     vadasg@gmail.com
#

import numpy
import Image
import os



def make_init(centerx,centery,side,imagefile,initfile,visualize=False):
    if 'png' != initfile[-3:].lower():
        #need to use png to avoid jpg artifacts
        raise NameError('Initialization file must be in png format.')

    #named labels for initialization file regions
    GC_BGD    = 0   #definitely background
    GC_FGD    = 1   #definitely foreground
    GC_PR_BGD = 2   #probably background
    GC_PR_FGD = 3   #probably foreground


    #load image and get dimensions
    img = Image.open(imagefile)
    w = img.size[0]
    h = img.size[1]


    #create array of the right type
    init = numpy.ones((w,h),numpy.uint8)
    init.shape = h,w


    #label regions
    #initially all background
    init *= GC_BGD

    #now label different regions of the image
    #this should be customized for the specific image set

    #big bounding box probably background
    init[centery-4*side:centery+4*side+1,centerx-4*side:centerx+4*side+1] = GC_PR_BGD

    #small bounding box probably foreground
    init[centery-side:centery+side+1,centerx-side:centerx+side+1] = GC_PR_FGD

    #eyetracking coordinate definitely foreground
    #make small square be foreground instead of just one point
    nudge = 3
    init[centery-nudge:centery+nudge+1,centerx-nudge:centerx+nudge+1] = GC_FGD
    #init[centery,centerx] = GC_FGD


    #display or save initialization file
    if visualize:
        import pylab
        pylab.imshow(init,origin='upper')
        pylab.colorbar()
        pylab.show()
    else:
        img2 = Image.fromarray(init,mode='L')
        img2.save(initfile)

def run_grabcut(imagefile, initfile, maskfile,numiter=4):
    cmd = './grabcut ' + str(numiter) + ' ' + imagefile + ' ' + initfile + ' ' + maskfile
    os.system(cmd)

def batch_process(directory,filetype='jpg',composite=True):
    #sample batch processing script


    import glob

    side = 25

    os.system('mkdir ./output')

    #files = glob.glob(directory + '/*.' + filetype)
    #for f in files:
        #imagefile = f


    positions = numpy.loadtxt('lanllive.txt',dtype=int)
    for p in positions:
        imagefile = directory + '/' + str(p[0]) + '.' + filetype
        centerx = p[1]
        centery = p[2]

        initfile = './output/init_mask_' + imagefile.split('/')[-1][:-3] + 'png'
        maskfile = './output/mask_' + imagefile.split('/')[-1]  #output file of grabcut
        print imagefile
        make_init(centerx,centery,side,imagefile,initfile,visualize=False)
        run_grabcut(imagefile, initfile, maskfile)

        if composite:  #make segmented composite images
            segmentedfile = './output/segmented_' + imagefile.split('/')[-1]
            os.system('composite ' + imagefile + ' -compose Multiply ' + maskfile + ' ' + segmentedfile)
            #os.system('rm -f ' + maskfile)

        os.system('rm -f ' + initfile)



if __name__ == '__main__':
    pass
    #batch_process('.')


    

