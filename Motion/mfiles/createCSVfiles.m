close all;
clc;

load = 0;

if(load==1)
    clear all;
    load=1;
end


%clip='030';

for(clpcnt=[26])
if(load==1)
    clear downsampledinput;
    clear mt1;
    clear mt2;
end

    clip=sprintf('%3.3d',clpcnt);
    %pvpfilepath='/Users/kpeterson/Documents/workspace/kris/output/3DGauss/';
    %pvpfilepath='/Users/kpeterson/Documents/competitionResults/Tower/output/';
    %pvpfilepath='/Users/kpeterson/Documents/competitionResults/Tower/output/050/050_005/';
    %pvpfilepath=['/Users/kpeterson/Documents/workdir/', clip, '/', clip, '_000/'];
    pvpfilepath=['/Volumes/Fantom/output/HeliNew/', clip, '/', clip, '_000/'];
    %pvpfilepath='/Volumes/Fantom/output/Tower/026/026_000/';
    %pvpfilepath='/Volumes/Time Machine Backups/stuff/outputHeli/045/045_000/';
    %pvpfilepath='/Users/kpeterson/Documents/competitionResults/Tower/output/052/052_005/';
    %pvpfilepath='/Users/kpeterson/Documents/competitionResults/Tower/output/051/tmp 6/output/';
    %pvpfilepath='/Users/kpeterson/Documents/Helo/';
    %pvpfilepath='/Users/kpeterson/Documents/Tailwind/output/';
    %pvpfilepath='/Users/kpeterson/Documents/testrun1st200/';
    %videopath='/Users/kpeterson/Documents/heli/050/';
    %videopath='/Volumes/Time Machine Backups/video/Tower-PNG/006/';
    %videopath='/Users/kpeterson/Documents/050/';
    %videopath='/Volumes/Time Machine Backups/video/Tower-PNG/001/';
    csvsavepath=pvpfilepath;
    %csvsavepath=['/Users/kpeterson/Documents/competitionData/neovision-results-challenge-heli/',clip, '/'];
    objecttype='Heli';


    videopath=pvpfilepath;
    mt1pvpnum=12; %27;
    multiplier=4;
    mtnumber=1;
    featurenum=2;
    %thresh=800; %tower
    thresh=50; %heli; %normally 14
    subclip='000';
    runcnt='002';
    display=1;
    displayinc=10;

    mkdir(pvpfilepath, 'generatedframes');

    oppfeaturenum=featurenum+2;
    
    if(load==1)
        downsampledinput=readactivities([pvpfilepath,'a2.pvp']);
    end
    if(load==1)
        mt1=readactivities([pvpfilepath, 'a', num2str(mt1pvpnum), '.pvp']);
        mt2=readactivities([pvpfilepath, 'a', num2str(mt1pvpnum+1), '.pvp']);
    end
    for(mtcnt=[1,2])
        mtnumber=mtcnt;
        %originalinput=readactivities([pvpfilepath,'a1.pvp']);
        %[origx, origy, orignf, origt]=size(originalinput);
        [xsize, ysize, fsize, tsize] = size(downsampledinput);
        origx = multiplier*xsize;
        origy = multiplier*ysize;
        orignf = fsize;
        origt = tsize;


        patchx=40;
        patchy=40;

        deltax=patchx/2;
        deltay=patchy/2;

        numpatchesx=int32(xsize/deltax);
        numpatchesy=int32(ysize/deltay);

        [xsize3, ysize3, fsize3, tsize] = size(mt2);
        %mt3=readactivities([pvpfilepath, 'a', num2str(mt1pvpnum+2), '.pvp']);


        %firstline='Frame,X1,Y1,X2,Y2,X3,Y3,X4,Y4,ObjectType,Occlusion,Ambiguous,Confidence,SiteInfo,Version,BB_X1,BB_Y1,BB_X2,BB_Y2,BB_X3,BB_Y3,BB_X4,BB_Y4';
        firstline='Frame,BB_X1,BB_Y1,BB_X2,BB_Y2,BB_X3,BB_Y3,BB_X4,BB_Y4,ObjectType,Occlusion,Ambiguous,Confidence,SiteInfo,Version';

        X1='0';Y1='0';X2=num2str(origx);Y2='0';X3='0';Y3=num2str(origy);X4=num2str(origx);Y4=num2str(origy);
        ObjectType='TARGET';
        occlusion='FALSE';
        ambiguous='FALSE';
        confidence='1.0';
        version='petavisionMT';

        %csvfilename=[objecttype, '_', clip, '_', subclip, '_PetaMotionMT', num2str(mtnumber), 'Feature', num2str(featurenum), '_Objects.csv'];
        csvfilename=[objecttype, '_', clip, '_PetaMotionMT', num2str(mtnumber), 'Feature', num2str(featurenum), '_Objects_', runcnt, '.csv'];
        fid=fopen([csvsavepath, csvfilename],'w+');
        %fid=1;
        fprintf(fid,'%s\n', firstline);

        %path='/Users/kpeterson/Documents/presentations/video/';
        %mt1=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/saveMTrunonCar/a39.pvp');
        %mt2=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/saveMTrunonCar/a40.pvp');
        %mt3=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/saveMTrunonCar/a41.pvp');
        %mt1=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a27.pvp');
        %mt2=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a28.pvp');
        %mt3=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a29.pvp');
        %dn=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/saveMTrunonCar/a2.pvp');
        %dn=readactivities('/Users/kpeterson/Documents/workspace/kris/output/3DGauss/a2.pvp');
        %figure(1)
        %stdblah= std(std(std(mt3(:,:,1,:))));
        %meanblah=mean(mean(mean(mt3(:,:,1,:))));
        %stdblah= std(std(std(mt1(:,:,1,:)-mt3(:,:,1,:))));
        %meanblah=mean(mean(mean(mt1(:,:,1,:)-mt3(:,:,1,:))));

        if(mtnumber==1)
            mtdiff=mt1(:,:,oppfeaturenum,[25:tsize])-mt1(:,:,featurenum,[25:tsize]);
        else
            mtdiff=mt2(:,:,oppfeaturenum,[25:tsize])-mt2(:,:,featurenum,[25:tsize]);
        end

        stdblah= std(std(std(mtdiff)));
        meanblah=mean(mean(mean(mtdiff)));
        mt0ff=6;
        threshhold=[];
        for(i=[25:tsize-mt0ff])
            %i
        %for(i=25)
            %X=figure(1);set(gcf,'Color','red');h=imagesc(double(squeeze(fliplr(rot90(rot90(dn(:,[1:250],1,i)))))));colormap(Gray);
            %temp=(squeeze(fliplr(rot90(rot90(rot90(downsampledinput(:,[1:250],1,i)))))));
            temp=(squeeze(fliplr(rot90(rot90(rot90(downsampledinput(:,:,1,i)))))));
            %temp=(squeeze((downsampledinput(:,[1:250],1,i))));
            im=uint8((temp/max(max(temp)))*128);
            [x, y] = size(im);

            %figure(2);%imagesc(double(squeeze(mt1(:,:,3,i)-mt1(:,:,1,i))));colormap(Gray);
            %blah=mt1(:,:,1,i)-mt3(:,:,1,i);
            if(mtnumber==1)
                mtdiff=mt1(:,:,oppfeaturenum,i+mt0ff)-mt1(:,:,featurenum,i+mt0ff);
            else
                mtdiff=mt2(:,:,oppfeaturenum,i+mt0ff)-mt2(:,:,featurenum,i+mt0ff);
            end

            blah=mtdiff;
            %blah2=max(max(blah))-blah;
            minblah=min(min(blah));
            blah2=blah-minblah;
            meanblahtmp=meanblah-minblah;

            threshhold=[threshhold,thresh*stdblah];
            %blah3=blah2<meanblahtmp-thresh*stdblah;
            blah3=blah2<meanblahtmp-0.7; %1.18; %1.2;  %0.9; %1.18
            %blah3=blah2>38;
            %imagesc(double(blah3(:,[1:250])));colormap(Gray);
            %set(h, 'AlphaData', ((1-fliplr(rot90(rot90(blah3(:,[1:250])))))))
            mask =  fliplr(rot90(rot90(rot90(blah3(:,:)))));
            %mask =  fliplr(rot90(rot90(rot90(blah3(:,[1:250])))));
            %mask =  blah3(:,[1:250]);
            invmask = 1-mask;
            %if(true)
            if((mod(i,displayinc)==0)&&(display==1))
                fig=figure(1); imagesc(im);colormap(gray)
            end

            for(xx=[1:numpatchesx])
                patchxloc=xx*deltax;
                patchmaxxloc=min(patchxloc+patchx,x);
                if(patchmaxxloc<=patchxloc) continue; end;
                for(yy=[1:numpatchesy])
                    patchyloc=yy*deltay;
                    patchmaxyloc=min(patchyloc+patchy,y);
                    if(patchmaxyloc<=patchyloc) continue; end;


                    patch=im([patchxloc:patchmaxxloc], [patchyloc:patchmaxyloc]);
                    patchmask=mask([patchxloc:patchmaxxloc], [patchyloc:patchmaxyloc]);

                    [rows cols]=find(patchmask);
                    if(~isempty(rows) && ~isempty(cols))
                        maxrow=max(rows);
                        minrow=min(rows);
                        maxcol=max(cols);
                        mincol=min(cols);
                        if((maxcol-mincol>5)&&(maxrow-minrow>5)&&(maxcol-mincol<3*patchy/4)&&(maxrow-minrow<3*patchx/4))
                        %if((maxcol-mincol<3*patchy/4)&&(maxrow-minrow<3*patchx/4))
                            boundingboxminyloc=patchyloc+mincol;
                            boundingboxmaxyloc=patchyloc+maxcol;
                            boundingboxminxloc=patchxloc+minrow;
                            boundingboxmaxxloc=patchxloc+maxrow;

                            %if(true)
                            if((mod(i,displayinc)==0)&&(display==1))


                                %figure;imagesc(patch); colormap(gray)
                                %drawing bounding box in blue
                                line([boundingboxminyloc boundingboxmaxyloc],[boundingboxmaxxloc boundingboxmaxxloc], 'LineWidth',4,'Color', 'b');
                                line([boundingboxminyloc boundingboxmaxyloc],[boundingboxminxloc boundingboxminxloc], 'LineWidth',4,'Color', 'b');
                                line([boundingboxmaxyloc boundingboxmaxyloc],[boundingboxminxloc boundingboxmaxxloc], 'LineWidth',4,'Color', 'b');
                                line([boundingboxminyloc boundingboxminyloc],[boundingboxminxloc boundingboxmaxxloc], 'LineWidth',4,'Color', 'b');

                                %drawing bounding box in green
                                line([patchyloc, patchmaxyloc],[patchmaxxloc, patchmaxxloc], 'LineWidth',4,'Color', 'g');
                                line([patchyloc, patchmaxyloc],[patchxloc patchxloc], 'LineWidth',4,'Color', 'g');
                                line([patchmaxyloc, patchmaxyloc],[patchxloc, patchmaxxloc], 'LineWidth',4,'Color', 'g');
                                line([patchyloc, patchyloc],[patchxloc, patchmaxxloc], 'LineWidth',4,'Color', 'g');

                            end

                            %fprintf(1,'%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d\n',
                            printstring='';
                            printstring = [num2str(i-4), ','];
        %                     printstring = [printstring, num2str(multiplier*(patchyloc-1)),',', num2str(multiplier*(patchxloc-1)),','];
        %                     printstring = [printstring, num2str(multiplier*(patchmaxyloc-1)),',',num2str(multiplier*(patchxloc-1)),','];
        %                     printstring = [printstring, num2str((multiplier*patchmaxyloc-1)),',',num2str(multiplier*(patchmaxxloc-1)),','];
        %                     printstring = [printstring, num2str(multiplier*(patchyloc-1)),',',num2str(multiplier*(patchmaxxloc-1)),','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxminyloc-35)), ',', num2str(multiplier*(boundingboxminxloc+1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxminyloc-35)), ',', num2str(multiplier*(boundingboxmaxxloc+1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxmaxyloc-35)), ',', num2str(multiplier*(boundingboxmaxxloc+1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxmaxyloc-35)), ',', num2str(multiplier*(boundingboxminxloc+1)), ','];
                            printstring = [printstring, num2str(multiplier*(boundingboxminyloc-1)), ',', num2str(multiplier*(boundingboxminxloc-1)), ','];
                            printstring = [printstring, num2str(multiplier*(boundingboxminyloc-1)), ',', num2str(multiplier*(boundingboxmaxxloc-1)), ','];
                            printstring = [printstring, num2str(multiplier*(boundingboxmaxyloc-1)), ',', num2str(multiplier*(boundingboxmaxxloc-1)), ','];
                            printstring = [printstring, num2str(multiplier*(boundingboxmaxyloc-1)), ',', num2str(multiplier*(boundingboxminxloc-1)), ','];
                            printstring = [printstring, ObjectType, ',', occlusion,',', ambiguous,',', confidence,','];
                            printstring = [printstring, ['File=', videopath,num2str(i,'%06.6d.png')], ',', version, '\n'];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxminyloc-1)), ',', num2str(multiplier*(boundingboxminxloc-1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxminyloc-1)), ',', num2str(multiplier*(boundingboxmaxxloc-1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxmaxyloc-1)), ',', num2str(multiplier*(boundingboxmaxxloc-1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxmaxyloc-1)), ',', num2str(multiplier*(boundingboxminxloc-1)), '\n'];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxminxloc-1)), ',', num2str(multiplier*(boundingboxminyloc-1)),','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxmaxxloc-1)), ',', num2str(multiplier*(boundingboxminyloc-1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxmaxxloc-1)), ',', num2str(multiplier*(boundingboxmaxyloc-1)), ','];
        %                     printstring = [printstring, num2str(multiplier*(boundingboxminxloc-1)), ',', num2str(multiplier*(boundingboxmaxyloc-1)), '\n'];
                            fprintf(fid,printstring);
                        end
                    end


                end
            end


            if((mod(i,displayinc)==0)&&(display==1))
            %if(true)
                filename=[pvpfilepath, 'generatedframes/', 'frameplusboxes', '_MT', num2str(mtnumber), '_', num2str(i,'%5.5i'),'.png'];
                saveas(fig,filename);
                %close(fig);
            end



        %     finalimage=zeros(x,y,3);
        %     finalimage(:,:,1)=im.*uint8(invmask) + uint8(mask*255);
        %     finalimage(:,:,2)=im.*uint8(invmask);
        %     finalimage(:,:,3)=im.*uint8(invmask);
        %     finalimage=(double(finalimage)/255);
        %     
        %     %whitebg('red')
        %     %i
        %     %figure(3);imagesc(double(squeeze(mt1(:,:,4,i)-mt1(:,:,2,i))));colormap(Gray);
        %     %pause
        %     %h=im2uint8(h)
        %     
        %     filename=[path, 'MTmatching_', num2str(i-19,'%5.5i'),'.png'];
            %imwrite(finalimage, filename);
            %figure(2); imagesc(finalimage)
            %pause
            %imwrite(h, filename);
            %saveas(gcf, filename);
            %saveas(X, filename);
            %hgsave(X, filename);
        end

        fclose(fid);
    end

end
mnthresh = mean(threshhold)
