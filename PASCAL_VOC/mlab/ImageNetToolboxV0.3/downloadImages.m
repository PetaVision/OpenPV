function downloadImages(homefolder, username, accesskey, wnid, isRecursive)  
   obtainServerMetaData(homefolder);
   if exist(fullfile(homefolder, 'DownloadStatus.xml'), 'file')
        docNode = xmlread(fullfile(homefolder, 'DownloadStatus.xml'));
        docRootNode = docNode.getDocumentElement;
        reply= checkResumeNeeded(docRootNode);
        if strcmp(reply, 'Y')
            resumeAndDownload(homefolder, username, accesskey, wnid, isRecursive, docNode);            
        else
            downloadNewSynsets(homefolder, username, accesskey, wnid, isRecursive, docNode)            
        end
    else 
         fprintf('Initializing new task...');
         downloadWNidList = genRecursiveSynsetList(homefolder, wnid, isRecursive);
         downloadWNidList = unique(downloadWNidList);
         genNewDownloadStatus(homefolder, downloadWNidList); 
         fprintf('OK\n');
         processTasklist(downloadWNidList, homefolder, username, accesskey);
    end    
end

function serverReleaseVersion = obtainServerMetaData(homefolder)
    StructureFileURL = 'http://www.image-net.org/api/xml/structure_released.xml';
    releaseStatusURL = 'http://www.image-net.org/api/xml/ReleaseStatus.xml';
    flag = 0;
    
    if ~exist(fullfile(homefolder, 'structure_released.xml'), 'file')
        fprintf('structure_released.xml does not exist, downloading the latest version from server...\n');
        urlwrite(StructureFileURL, fullfile(homefolder, 'structure_released.xml'));
        fprintf('finished\n');
        flag = 1;
    end
    serverReleaseVersion = requestServerReleaseVersion();    
    if exist(fullfile(homefolder, 'ReleaseStatus.xml'), 'file')         
        releaseDocNode = xmlread(fullfile(homefolder, 'ReleaseStatus.xml'));
        releaseDocRootNode = releaseDocNode.getDocumentElement;
        query = sprintf('/ReleaseStatus/releaseData');
        result = XPathExecuteQuery(releaseDocRootNode, query);
        clientReleaseVersion = char(result.item(0).getFirstChild.getNodeValue);
        if ~strcmp(serverReleaseVersion, clientReleaseVersion)
            fprintf('Downloading latest ReleaseStatus.xml from server...\n');
            urlwrite(releaseStatusURL, fullfile(homefolder, 'ReleaseStatus.xml'));
            if flag ~= 1 
                fprintf('Downloading the latest structure_released.xml from server...\n');
                urlwrite(StructureFileURL, fullfile(homefolder, 'structure_released.xml'));
            end;
            fprintf('finished\n');
        end    
    else
        fprintf('Downloading ReleaseStatus.xml from server...\n');
        urlwrite(releaseStatusURL, fullfile(homefolder, 'ReleaseStatus.xml'));
        fprintf('finished\n');
    end   
end

function resumeAndDownload(homefolder, username, accesskey, wnid, isRecursive, docNode)
    resumeUnfinishedSynsts(homefolder, username, accesskey, docNode);
    downloadNewSynsets(homefolder, username, accesskey, wnid, isRecursive, docNode)            
end

function resumeUnfinishedSynsts(homefolder, username, accesskey, docNode)  
    fprintf('Resume downloading...\n');
    docRootNode = docNode.getDocumentElement;
    downloadWNidList = getResumeDownloadList(docRootNode);
    processTasklist(downloadWNidList, homefolder, username, accesskey);
end

function downloadNewSynsets(homefolder, username, accesskey, wnid, isRecursive, docNode)
    % % % % % 
    % here add a function whether to remove the resume list             
    % % % % %       
    downloadWNidList = genRecursiveSynsetList(homefolder, wnid, isRecursive);          
    downloadWNidList = unique(downloadWNidList);
    checkWNidList(homefolder, downloadWNidList, docNode);
    processTasklist(downloadWNidList, homefolder, username, accesskey);
end

function  wnidList = genRecursiveSynsetList(homefolder, wnid, isRecursive) 
    xDoc = xmlread(fullfile(homefolder, 'structure_released.xml'));
    if isRecursive
        if strcmp(wnid, 'gproot')
            query = sprintf('/ImageNetStructure//synset[@wnid="%s"]/descendant::synset', wnid);
        else
            query = sprintf('/ImageNetStructure//synset[@wnid="%s"]/descendant-or-self::synset', wnid);
        end
    else
        query = sprintf('/ImageNetStructure//synset[@wnid="%s"]', wnid);
    end
    result = XPathExecuteQuery(xDoc, query);
    % used to deduplicate the same synset returned multiple times
    query2 = sprintf('/ImageNetStructure//synset[@wnid="%s"]', wnid);
    r = XPathExecuteQuery(xDoc, query2);
    nOccurance = r.getLength();
    
     if (result.getLength() == 0) || (nOccurance == 0)
        error('\nERROR ! %s is not found in structure_released.xml\n Please check your synset WordNet ID !', wnid);
    else
        wnidList = cell(1, result.getLength()/nOccurance);
     end
    
    for i = 0 : result.getLength()/nOccurance - 1
         wnid = char(result.item(i).getAttribute('wnid'));
         wnidList{(i+1)} = wnid;
    end
end

function genNewDownloadStatus(homefolder, wnidList) 
    docNode = com.mathworks.xml.XMLUtils.createDocument('DownloadStatus');
    length = size(wnidList, 2);
    for i = 0 : length - 1
        wnid= wnidList{(i + 1)};
        addNewNode(homefolder, docNode, wnid);
    end
end

function addNewNode(homefolder, docNode, wnid) 
    synsetTaskNode = docNode.createElement('synsetTask');
    docRootNode = docNode.getDocumentElement;
    docRootNode.appendChild(synsetTaskNode);

    wnidNode = docNode.createElement('wnid');
    synsetTaskNode.appendChild(wnidNode);
    wnidNode.appendChild(docNode.createTextNode(wnid));

    completedNode = docNode.createElement('completed');
    synsetTaskNode.appendChild(completedNode);
    completedNode.appendChild(docNode.createTextNode('0'));

    versionNode = docNode.createElement('version');
    synsetTaskNode.appendChild(versionNode);
    versionNode.appendChild(docNode.createTextNode('Uninitialized'));

    storePathNode = docNode.createElement('storePath');
    synsetTaskNode.appendChild(storePathNode);
    storePathNode.appendChild(docNode.createTextNode(homefolder));
    
    xmlwrite(fullfile(homefolder, 'DownloadStatus.xml'), docNode);
end

function checkWNidList(homefolder, wnidList, docNode)

    ht = buildClientHashtable(homefolder, 'wnid', 'completed');
     
    length = size(wnidList, 2);
    for i = 0 : length - 1
        key = char(wnidList(i + 1));
        if ht.containsKey(key)
            if strcmp(ht.get(key), '1')
                %%%%%%%%
                %remind user synset has been downloaded before, Y to
                %redownload, N to cancel
                %%%%%%%%
                fprintf('The synset %s has been downloaded before, prepare to re-download...\n', key);
                
                query = sprintf('/DownloadStatus/synsetTask[wnid = "%s"]/completed', key);
                completedResult = XPathExecuteQuery(docNode.getDocumentElement, query);
                
                query = sprintf('/DownloadStatus/synsetTask[wnid = "%s"]/version', key);
                versionResult = XPathExecuteQuery(docNode.getDocumentElement, query);
                
                if completedResult.getLength() > 0
                    completedItem = completedResult.item(0);
                    completedItem.getFirstChild.setNodeValue('0');
                    versionItem = versionResult.item(0);
                    versionItem.getFirstChild.setNodeValue('Uninitialized');
                else 
                    error('DownloadStatus.xml error ! %s', wnid);
                end
                
                xmlwrite(fullfile(homefolder, 'DownloadStatus.xml'), docNode);
            else
                fprintf('The wnid  %s is in the unfinished list \n', key);
            end
        else
            fprintf('Add new synsetTask %s to DownloadStatus.xml\n', key);
            addNewNode(homefolder, docNode, key);
        end
    end
end
