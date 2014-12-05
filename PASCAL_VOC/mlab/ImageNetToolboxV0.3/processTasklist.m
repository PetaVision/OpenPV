function processTasklist(wnidList, homefolder, username, accesskey)
    msg = verifyUserAccesskeyPair(username, accesskey);
    if ~strcmp(msg, 'OK')
        if strcmp(msg, 'WRONG_USER_ACCESSKEY_PAIR')
            error('User does not exist or your username and accesskey do not match!');
        elseif strcmp(msg, 'PERMISSION_NOT_GRANTED')
            error('You have not been granted the permission to access full resolution images!');
        else
            error('You username and accesskey verification failed!');
        end        
    end
    
    ServerURL = 'http://www.image-net.org';
    docNode = xmlread(fullfile(homefolder,'DownloadStatus.xml'));
    docRootNode = docNode.getDocumentElement;
    
%     releaseStatusURL = 'http://www.image-net.org/api/xml/ReleaseStatus.xml';
%    serverHashtable = java.util.Hashtable;
   
%     if exist(fullfile(homefolder, 'ReleaseStatus.xml'), 'file') 
%         serverReleaseVersion = requestServerReleaseVersion();
%         releaseDocNode = xmlread(fullfile(homefolder, 'ReleaseStatus.xml'));
%         releaseDocRootNode = releaseDocNode.getDocumentElement;
%         query = sprintf('/ReleaseStatus/releaseData');
%         result = XPathExecuteQuery(releaseDocRootNode, query);
%         clientReleaseVersion = char(result.item(0).getFirstChild.getNodeValue);
%         if ~strcmp(serverReleaseVersion, clientReleaseVersion)
%             fprintf('Downloading latest ReleaseStatus.xml from server...\n');
%             urlwrite(releaseStatusURL, homefolder);
%             fprintf('Downloading latest structure.xml from server...\n');
%             urlwrite(StructureFileURL, fullfile(homefolder, 'structure.xml'));
%             fprintf('finished\n');
%         end    
%     else
%         fprintf('Downloading ReleaseStatus.xml from server...\n');
%         urlwrite(releaseStatusURL, fullfile(homefolder, 'ReleaseStatus.xml'));
%         fprintf('finished\n');
%     end
    
    serverHashtable = serverReleaseStatusToHash(homefolder);

    length = size(wnidList, 2);
    
    for i = 0 : length - 1
        wnid = wnidList{(i + 1)};
        
        if ~serverHashtable.containsKey(wnid)
            fprintf('skip %s...\n  Your query is not available yet. ImageNet is still under construction.\n  Please visit www.image-net.org to check what we have\n', wnid);
            continue;
        end
        
        serverReleaseVersion =  serverHashtable.get(wnid);
        downloadURL = sprintf('%s/download/synset?wnid=%s&username=%s&accesskey=%s&release=%s', ServerURL, wnid, username, accesskey, serverReleaseVersion);
        fprintf(sprintf('Downloading synset %s...\n', wnid));
        dstTarPath = fullfile(homefolder, [wnid, '.tar']);
        urlwrite(downloadURL, dstTarPath);

        query = sprintf('/DownloadStatus/synsetTask[wnid = "%s"]/version', wnid);
        versionResult = XPathExecuteQuery(docRootNode, query);
        if versionResult.getLength() > 0            
            versionResult.item(0).getFirstChild.setNodeValue(serverReleaseVersion);
        else 
            fprintf('downloadStatus error ! %s', wnid);
        end
        
        query = sprintf('/DownloadStatus/synsetTask[wnid = "%s"]/completed', wnid);
        completedResult = XPathExecuteQuery(docRootNode, query);
        if completedResult.getLength() > 0
            completedItem = completedResult.item(0);
            completedItem.getFirstChild.setNodeValue('1');
        else 
            error('DownloadStatus.xml error ! %s', wnid);
        end
        
         xmlwrite(fullfile(homefolder, 'DownloadStatus.xml'), docNode);
    end
     
end
