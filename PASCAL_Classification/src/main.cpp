#include <stdio.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <columns/buildandrun.hpp>
#include <layers/ImageFromMemoryBuffer.hpp>
#include "cMakeHeader.h"

#define TEXTFILEBUFFERSIZE 1024

int parseConfigFile(char ** imageLayerNamePtr, char ** resultLayerNamePtr, char ** octaveCommandPtr, char ** heatMapMontageDirPtr);
int parseConfigParameter(char const * inputLine, char const * configParameter, char ** parameterPtr, unsigned int lineNumber);
char * getImageFileName(InterColComm * icComm);

int main(int argc, char* argv[])
{
    int status = PV_SUCCESS;

    // clobber octave logfile; octave output will be appended to this file.
    char const * octavelogfile = "octave.log";
    FILE * octavefp = fopen(octavelogfile, "w");
    fclose(octavefp);

    int octavepid = 0; // pid of the child octave process.

    // Build the column from the params file
    PV::HyPerCol * hc = build(argc, argv);

    // Parse config file for image layer, result layer, file of image files
    char * imageLayerName = NULL;
    char * resultLayerName = NULL;
    char * octaveCommand = NULL;
    char * heatMapMontageDir = NULL;
    if (hc->columnId()==0)
    {
        status = parseConfigFile(&imageLayerName, &resultLayerName, &octaveCommand, &heatMapMontageDir);
    }
#ifdef PV_USE_MPI
    MPI_Bcast(&status, 1/*count*/, MPI_INT, 0/*rootprocess*/, hc->icCommunicator()->communicator());
#endif // PV_USE_MPI
    if (status != PV_SUCCESS) { MPI_Finalize(); exit(EXIT_FAILURE); }

    BaseLayer * imageBaseLayer = hc->getLayerFromName(imageLayerName);
    if (imageBaseLayer==NULL)
    {
        if (hc->columnId()==0)
        {
            fprintf(stderr, "%s error: no layer matches imageLayerName = \"%s\"\n", argv[0], imageLayerName);
        }
        status = PV_FAILURE;
    }
    ImageFromMemoryBuffer * imageLayer = dynamic_cast<ImageFromMemoryBuffer *>(imageBaseLayer);
    if (imageLayer==NULL)
    {
        if (hc->columnId()==0)
        {
            fprintf(stderr, "%s error: imageLayerName = \"%s\" is not an ImageFromMemoryBuffer layer\n", argv[0], imageLayerName);
        }
        status = PV_FAILURE;
    }

    BaseLayer * resultBaseLayer = hc->getLayerFromName(resultLayerName);
    if (resultBaseLayer==NULL)
    {
        if (hc->columnId()==0)
        {
            fprintf(stderr, "%s error: no layer matches resultLayerName = \"%s\"\n", argv[0], resultLayerName);
        }
        status = PV_FAILURE;
    }
    HyPerLayer * resultLayer = dynamic_cast<HyPerLayer *>(resultBaseLayer);
    if (resultLayer==NULL)
    {
        if (hc->columnId()==0)
        {
            fprintf(stderr, "%s error: resultLayerName = \"%s\" is not a HyPerLayer\n", argv[0], resultLayerName);
        }
        status = PV_FAILURE;
    }

    char const * imagePvpFile = imageLayer->getOutputStatePath();
    char const * resultPvpFile = resultLayer->getOutputStatePath();

#ifdef PV_USE_MPI
    MPI_Bcast(&status, 1/*count*/, MPI_INT, 0/*rootprocess*/, hc->icCommunicator()->communicator());
#endif // PV_USE_MPI
    if (status != PV_SUCCESS) { MPI_Finalize(); exit(EXIT_FAILURE); }

    int layerNx = imageLayer->getLayerLoc()->nxGlobal;
    int layerNy = imageLayer->getLayerLoc()->nyGlobal;
    int layerNf = imageLayer->getLayerLoc()->nf;

    int imageNx = layerNx;
    int imageNy = layerNy;
    int imageNf = 3;

    int bufferNx = layerNx;
    int bufferNy = layerNy;
    int bufferNf = imageNf;

    size_t imageBufferSize = (size_t)bufferNx*(size_t)bufferNy*(size_t)bufferNf;
    uint8_t * imageBuffer = NULL;
    if (hc->columnId()==0)
    {
        GDALAllRegister();
    }

    double startTime = hc->getStartTime();
    double stopTime = hc->getStopTime();
    double dt = hc->getDeltaTime();
    double displayPeriod = stopTime - startTime;

    uint8_t zeroVal = (uint8_t) 0;
    uint8_t oneVal = (uint8_t) 255;

    if (hc->columnId()==0) {
        struct stat heatMapMontageStat;
        status = stat(heatMapMontageDir, &heatMapMontageStat);
        if (status!=0 && errno==ENOENT) {
            status = mkdir(heatMapMontageDir, 0770);
            if (status!=0) {
                fprintf(stderr, "Error: Unable to make heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
                exit(EXIT_FAILURE);
            }
            status = stat(heatMapMontageDir, &heatMapMontageStat);
        }
        if (status!=0) {
            fprintf(stderr, "Error: Unable to get status of heat map montage directory \"%s\": %s\n", heatMapMontageDir, strerror(errno));
            exit(EXIT_FAILURE);
        }
        if (!(heatMapMontageStat.st_mode & S_IFDIR)) {
            fprintf(stderr, "Error: Heat map montage \"%s\" is not a directory\n", heatMapMontageDir);
            exit(EXIT_FAILURE);
        }
    }
    

    // Main loop: get an image, load it into the image layer, do HyPerCol::run, lather, rinse, repeat
    char * imageFile = getImageFileName(hc->icCommunicator());
    while(imageFile!=NULL && imageFile[0]!='\0')
    {
        if (hc->columnId()==0)
        {
            printf("Image \"%s\"\n", imageFile);
            GDALDataset * gdalDataset = PV_GDALOpen(imageFile);
            if (gdalDataset==NULL)
            {
                fprintf(stderr, "%s: GDALOpen failed for image \"%s\".\n", argv[0], imageFile);
                exit(EXIT_FAILURE);
            }
            int imageNx= gdalDataset->GetRasterXSize();
            int imageNy = gdalDataset->GetRasterYSize();
            int imageNf = GDALGetRasterCount(gdalDataset);
            // Need to rescale so that the the short side of the image equals the short side of the layer
            // ImageFromMemoryBuffer layer will handle the cropping.
            double xScaleFactor = (double)layerNx / (double) imageNx;
            double yScaleFactor = (double)layerNy / (double) imageNy;
            if (xScaleFactor < yScaleFactor) /* need to rescale so that bufferNy=layerNy and bufferNx>layerNx */
            {
                bufferNx = (int) round(imageNx * yScaleFactor);
                bufferNy = layerNy;
            }
            else {
                bufferNx = layerNx;
                bufferNy = (int) round(imageNy * xScaleFactor);
            }
            bufferNf = layerNf;
            size_t newImageBufferSize = (size_t)bufferNx * (size_t)bufferNy * (size_t)bufferNf;
            if (imageBuffer==NULL || newImageBufferSize != imageBufferSize)
            {
                imageBufferSize = newImageBufferSize;
                imageBuffer = (uint8_t *) realloc(imageBuffer, imageBufferSize*sizeof(uint8_t));
                if (imageBuffer==NULL)
                {
                    fprintf(stderr, "%s: Unable to resize image buffer to %d-by-%d-by-%d for image \"%s\": %s\n",
                             argv[0], bufferNx, bufferNy, bufferNf, imageFile, strerror(errno));
                    exit(EXIT_FAILURE);
                }
            }

            bool isBinary = true;
            for (int iBand=0;iBand<imageNf; iBand++)
            {
                GDALRasterBandH hBand = GDALGetRasterBand(gdalDataset,iBand+1);
                char ** metadata = GDALGetMetadata(hBand, "Image_Structure");
                if(CSLCount(metadata) > 0){
                    bool found = false;
                    for(int i = 0; metadata[i] != NULL; i++){
                        if(strcmp(metadata[i], "NBITS=1") == 0){
                           found = true;
                           isBinary &= true;
                           break;
                        }
                    }
                    if(!found){
                       isBinary &= false;
                    }
                }
                else{
                    isBinary = false;
                }
                GDALDataType dataType = gdalDataset->GetRasterBand(iBand+1)->GetRasterDataType(); // Why are we using both GDALGetRasterBand and GDALDataset::GetRasterBand?
                if (dataType != GDT_Byte)
                {
                    fprintf(stderr, "%s error: Image file \"%s\", band %d, is not GDT_Byte type.\n", argv[0], imageFile, iBand+1);
                    exit(EXIT_FAILURE);
                }
            }

#ifdef PV_USE_OPENMP_THREADS
#pragma omp parallel for
#endif
            for (size_t n=0; n < imageBufferSize; n++)
            {
                imageBuffer[n] = oneVal;
            }

            int xStride = bufferNf;
            int yStride = bufferNf * bufferNx;
            int bandStride = 1;

            gdalDataset->RasterIO(GF_Read, 0/*xOffset*/, 0/*yOffset*/, imageNx, imageNy, imageBuffer, bufferNx, bufferNy,
                    GDT_Byte, layerNf, NULL, xStride*sizeof(uint8_t), yStride*sizeof(uint8_t), bandStride*sizeof(uint8_t));

            imageLayer->setMemoryBuffer(imageBuffer, bufferNy, bufferNx, bufferNf, xStride, yStride, bandStride, zeroVal, oneVal);
            GDALClose(gdalDataset);
        }
        startTime = hc->simulationTime();
        stopTime = startTime + displayPeriod;
        hc->run(startTime, stopTime, dt);

        if (hc->columnId()==0)
        {
            fflush(imageLayer->clayer->activeFP->fp);
            fflush(resultLayer->clayer->activeFP->fp);
        }
        else
        {
            assert(imageLayer->clayer->activeFP==NULL);
            assert(resultLayer->clayer->activeFP==NULL);
        }
        int numParams = 20;
        int params[numParams];

        PV_Stream * imagePvpStream = PV_fopen(imagePvpFile, "r", false/*verifyWrites*/);
        status = pvp_read_header(imagePvpStream, hc->icCommunicator(), params, &numParams);
        if (status!=PV_SUCCESS)
        {
            fprintf(stderr, "pvp_read_header for imageLayer \"%s\" outputfile \"%s\" failed.\n", imageLayer->getName(), imagePvpFile);
            exit(EXIT_FAILURE);
        }
        PV_fclose(imagePvpStream);
        assert(numParams==20);
        int imageFrameNumber = params[INDEX_NBANDS];

        PV_Stream * resultPvpStream = PV_fopen(resultPvpFile, "r", false/*verifyWrites*/);
        status = pvp_read_header(resultPvpStream, hc->icCommunicator(), params, &numParams);
        if (status!=PV_SUCCESS)
        {
            fprintf(stderr, "pvp_read_header for resultLayer \"%s\" outputfile \"%s\" failed.\n", resultLayer->getName(), resultPvpFile);
            exit(EXIT_FAILURE);
        }
        PV_fclose(resultPvpStream);
        assert(numParams==20);
        int resultFrameNumber = params[INDEX_NBANDS];
        std::stringstream montagePath("");
        montagePath << heatMapMontageDir << "/frame" << imageFrameNumber << ".png";

        free(imageFile);
        imageFile = getImageFileName(hc->icCommunicator());

        if (hc->columnId()==0)
        {
            if (octavepid>0)
            {
                int waitprocess;
                int waitstatus = waitpid(octavepid, &waitprocess, 0);
                if (waitstatus < 0)
                {
                    fprintf(stderr, "waitpid failed: %s\n", strerror(errno));
                    exit(EXIT_FAILURE);
                }
                octavepid = 0;
            }
            fflush(stdout); // so that unflushed buffer isn't copied to child process
            octavepid = fork();
            if (octavepid < 0)
            {
                fprintf(stderr, "fork() error: %s\n", strerror(errno));
                exit(EXIT_FAILURE);
            }
            else if (octavepid==0) {
                /* child process */
                std::stringstream octavecommandstream("");
                octavecommandstream << "octave --eval 'heatMapMontage(\"" <<
                        imagePvpFile << "\", \"" <<
                        resultPvpFile << "\"," <<
                        imageFrameNumber << ", " <<
                        resultFrameNumber << ", " <<
                        "\"" << montagePath.str() << "\"" << ");'" <<
                        " >> " << octavelogfile << " 2>&1";
                std::ofstream octavelogstream;
                octavelogstream.open(octavelogfile, std::fstream::out | std::fstream::app);
                octavelogstream << "Calling octave with the command\n";
                octavelogstream << octavecommandstream.str() << "\n";
                octavelogstream.close();
                system(octavecommandstream.str().c_str()); // Analysis of the result of the current frame
                exit(EXIT_SUCCESS);
            }
        }
    }

    delete hc;
    free(imageFile);
    free(imageLayerName);
    free(resultLayerName);

    return status==PV_SUCCESS ? EXIT_SUCCESS : EXIT_FAILURE;   
}

int parseConfigFile(char ** imageLayerNamePtr, char ** resultLayerNamePtr, char ** octaveCommandPtr, char ** heatMapMontageDirPtr)
{
    unsigned int const numConfigParameters = 4;
    // This routine should only be called by the root process
    int status = PV_SUCCESS;
    FILE * parseConfigFileFP = fopen(CONFIG_FILE, "r");
    if (parseConfigFileFP == NULL)
    {
        fprintf(stderr, "Unable to open config file \"%s\": %s\n", CONFIG_FILE, strerror(errno));
        return PV_FAILURE;
    }
    *imageLayerNamePtr = NULL;
    *resultLayerNamePtr = NULL;
    *octaveCommandPtr = NULL;
    *heatMapMontageDirPtr = NULL;
    char imagebuffer[TEXTFILEBUFFERSIZE];
    unsigned int linenumber=0;
    unsigned int configParametersRead = 0;
    while (configParametersRead < numConfigParameters)
    {
        linenumber++;
        char * line = fgets(imagebuffer, TEXTFILEBUFFERSIZE, parseConfigFileFP);
        if (line==NULL) { break; }
        char * colonsep = strchr(line,':');
        if (colonsep==NULL) { break; }
        char * openquote = strchr(colonsep,'"');
        if (openquote==NULL) { break; }
        char * closequote = strchr(openquote+1,'"');
        if (closequote==NULL) { break; }
        *colonsep='\0';
        *openquote='\0';
        *closequote='\0';
        char * keyword = line;
        char * value = &openquote[1];

        if (!strcmp(line,"imageLayer"))
        {
            status = parseConfigParameter(line, value, imageLayerNamePtr, linenumber);
            if (status != PV_SUCCESS) { break; }
            configParametersRead++;
        }
        if (!strcmp(line,"resultLayer"))
        {
            status = parseConfigParameter(line, value, resultLayerNamePtr, linenumber);
            if (status != PV_SUCCESS) { break; }
            configParametersRead++;
        }
        if (!strcmp(line,"octaveCommand"))
        {
            status = parseConfigParameter(line, value, octaveCommandPtr, linenumber);
            if (status != PV_SUCCESS) { break; }
            configParametersRead++;
        }
        if (!strcmp(line,"heatMapMontageDir"))
        {
            status = parseConfigParameter(line, value, heatMapMontageDirPtr, linenumber);
            if (status != PV_SUCCESS) { break; }
            configParametersRead++;
        }
    }
    if (*imageLayerNamePtr==NULL)
    {
        fprintf(stderr, "imageLayer was not defined in %s.\n", CONFIG_FILE);
        status = PV_FAILURE;
    }
    if (*resultLayerNamePtr==NULL)
    {
        fprintf(stderr, "resultLayer was not defined in %s.\n", CONFIG_FILE);
        status = PV_FAILURE;
    }
    if (*octaveCommandPtr==NULL)
    {
        fprintf(stderr, "octaveCommand was not defined in %s.\n", CONFIG_FILE);
        status = PV_FAILURE;
    }
    if (*heatMapMontageDirPtr==NULL)
    {
        fprintf(stderr, "heatMapMontageDir was not defined in %s.\n", CONFIG_FILE);
        status = PV_FAILURE;
    }
    fclose(parseConfigFileFP);
    return status;
}

int parseConfigParameter(char const * configParameter, char const * configValue, char ** parameterPtr, unsigned int lineNumber)
{
    if (*parameterPtr != NULL)
    {
        fprintf(stderr, "Line %u: Multiple lines defining %s: already set to \"%s\"; duplicate value is \"%s\"\n", lineNumber, configParameter, *parameterPtr, configValue);
        return PV_FAILURE;
    }
    *parameterPtr = strdup(configValue);
    if (*parameterPtr == NULL)
    {
        fprintf(stderr, "Error setting %s from config file: %s\n", configParameter, strerror(errno));
        return PV_FAILURE;
    }
    printf("%s set to %s\n", configParameter, configValue);
    return PV_SUCCESS;
}

char * getImageFileName(InterColComm * icComm)
{
    // All processes call this routine.  Calling routine is responsible for freeing the returned string.
    char buffer[TEXTFILEBUFFERSIZE];
    int rank=icComm->commRank();
    if (rank==0)
    {
        bool found = false;
        while(!found)
        {
            printf("Enter filename: "); fflush(stdout);
            char * result = fgets(buffer, TEXTFILEBUFFERSIZE, stdin);
            if (result==NULL) { break; }

            // Ignore lines containing only whitespace
            for (char const * c = result; *c; c++)
            {
                if (!isspace(*c)) { found=true; break; }
            }
        }
        if (found)
        {
            size_t len = strlen(buffer);
            assert(len>0);
            if (buffer[len-1]=='\n') { buffer[len-1]='\0'; }
        }
        else
        {
            buffer[0] = '\0';
        }
    }
#ifdef PV_USE_MPI
    MPI_Bcast(buffer, TEXTFILEBUFFERSIZE/*count*/, MPI_CHAR, 0/*rootprocess*/, icComm->communicator());
#endif // PV_USE_MPI
    char * filename = strdup(buffer);
    if (filename==NULL)
    {
        fprintf(stderr, "Rank %d process unable to allocate space for line from listOfImageFiles file.\n", rank);
        exit(EXIT_FAILURE);
    }
    return filename;
}
