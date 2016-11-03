#include "MultimodalVideoRegistrAlg.h"
#include "DatasetUtils.h"

// minimum foreground blob area required for contour point matching
#define MIN_BLOB_AREA                   100
// approximative frame size of the correspondance buffer
#define NB_FRAME_MEMORY                 100
// approximative contour point count in a single frame (used for init only)
#define KEYPTS_PER_FRAME_BUFFER         100
// minimum number of contour points required for basic transformation estimation
#define MIN_KEYPOINTS_COUNT             5//9
// minimum number of contour points required for TPS model estimation
#define MIN_TPS_KEYPOINTS_COUNT         25//29
// minimum blob shape match similarity score required to add contour points to buffer
#define MIN_SIMILARITY_SCORE            1.0f
// initial/default value for smoothing count variable ('alpha' in the paper)
#define DEFAULT_SMOOTH_COUNT            2
// defines whether to match shapes frame-wide or blob-per-blob
#define USE_PER_BLOB_SHAPE_DIST         1
// defines whether to invert the FG region size ratio test or not
#define INVERT_FG_PROPORTION            0

MultimodalVideoRegistrAlg::MultimodalVideoRegistrAlg() {
    ResetInternalState();
}

void MultimodalVideoRegistrAlg::ResetInternalState() {
    m_pShapeDistExtr = cv::createShapeContextDistanceExtractor(15,5,0.1f,4.0f);
    m_pShapeTransf_TPS = cv::createThinPlateSplineShapeTransformer();
    m_pShapeTransf_Aff = cv::createAffineTransformer(false);
    m_vvoRecentFiltKeyPoints_ToTransform = std::vector<std::vector<cv::Point> >(NB_FRAME_MEMORY);
    m_vvoRecentFiltKeyPoints = std::vector<std::vector<cv::Point> >(NB_FRAME_MEMORY);
    m_nSmoothingCount = DEFAULT_SMOOTH_COUNT;
    m_bLoopedAtLeastOnce = false;
    m_nCurrFrameStackIdx = 0;
    m_nCurrModelNextCoordIdx = 2;
    m_nCurrModelLastCoordIdx = 0;
    m_fPrevBestForegroundOverlapError = 1.0f;
    m_vfModelCoords_ToTransform = std::vector<float>(NB_FRAME_MEMORY*KEYPTS_PER_FRAME_BUFFER,0.0f);
    m_vfModelCoords = std::vector<float>(NB_FRAME_MEMORY*KEYPTS_PER_FRAME_BUFFER,0.0f);
    m_vnModelCoordsPersistence = std::vector<int>(m_vfModelCoords_ToTransform.size()/2,0);
    m_oLatestTransMat = cv::Mat();
    m_oLatestTransMat_inv = cv::Mat();
    m_oBestTransMat = cv::Mat::eye(3,3,CV_64FC1);
    m_oBestTransMat_inv = cv::Mat::eye(3,3,CV_64FC1);
    m_oCurrTransMat = cv::Mat();
    m_oCurrTransMat_inv = cv::Mat();

	//New one
	m_bIsTheSame = true;
	m_iFrameCountDown = 10;

	m_curOverlapError = INT_MAX;
	minError = INT_MAX;

	m_oBestChooseTransMat = cv::Mat::eye(3, 3, CV_64FC1);
	m_oBestChooseTransMat_inv = cv::Mat::eye(3, 3, CV_64FC1);

	iCountFrame = -1;

	iIndexCurrentFrame = 0;

	m_isPreProcessed = false;

	//m_oBestFundamentalMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	bestTransFundMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	fCurFundamentalMatrixOverlapError_Best = 1.0;
	isUsingFundametalMatrix = false;

	minisBad = minisBadFundamental = 100;

	m_ratioH1H2 = 1;

	m_ErrorH1H2Cur = m_ErrorH1H2Best = 1;

	isRunSecond = false;
}


void MultimodalVideoRegistrAlg::reduceSize(std::vector<std::vector<cv::Point> >& v){
	int tmp = v.size();
	bool bGo = false;
	newVec = new std::vector<std::vector<cv::Point> >();
	(*newVec).resize(tmp);
	for (int i = 0; i < tmp; i++){
		if (v[i].size() < 40)
			continue;
		bGo = true;

		for (int j = 0; j < v[i].size(); j += 2)
			(*newVec)[i].push_back(v[i][j]);
	}

	if (bGo == true)
		v = *newVec;

	delete newVec;
}




void MultimodalVideoRegistrAlg::ProcessForeground(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground, bool isCalculatingFundamental) {
	 
	m_List.push_back(make_pair(oForeground_ToTransform.clone(), oForeground.clone()));
	AdaptiveFrame(oForeground_ToTransform, oForeground, 1);

	if (isCalculatingFundamental == true){
		//PreProcessForeground(oForeground_ToTransform, oForeground);
	}

	iIndexCurrentFrame++;
	cv::Mat oForeground_ToTransform_tmp = oForeground_ToTransform.clone();
	cv::Mat oForeground_tmp = oForeground.clone();


	//Shape extraction STEP!!
	cv::findContours(oForeground_ToTransform_tmp, m_vvoBlobContours_ToTransform, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(oForeground_tmp, m_vvoBlobContours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	
	//imshow("oForeground", oForeground);
	//imshow("oForeground_ToTransform", oForeground_ToTransform);

    if(!m_vvoBlobContours_ToTransform.empty() && !m_vvoBlobContours.empty()) {
		


		//for (int i = 0; i<m_vvoBlobContours.size(); i++)
		//	cout << "Visual= " << m_vvoBlobContours[i].size() << " ";
		//cout << endl;
		//for (int i = 0; i<m_vvoBlobContours_ToTransform.size(); i++)
		//	cout << "Thermal= " << m_vvoBlobContours_ToTransform[i].size() << " ";
		//cout << endl;

		//My skills
		reduceSize(m_vvoBlobContours_ToTransform);
		reduceSize(m_vvoBlobContours);
		m_iFrameCountDown = m_iFrameCountDown > 0 ? m_iFrameCountDown-1 : 0;


		//cout << m_iFrameCountDown << " " << m_bIsTheSame << endl;
		if (m_iFrameCountDown == 0)
			m_bIsTheSame = true;

        std::vector<std::vector<cv::Point> > vvoFiltBlobContours_ToTransform, vvoFiltBlobContours;
        std::vector<std::vector<cv::DMatch> > vvoFiltBlobKeyPointsMatches;

		if (m_vArrayMatchContours.empty())
			m_vArrayMatchContours.resize(m_vvoBlobContours_ToTransform.size(), -1);

		if (m_vvoBlobContours.size() != m_vvoBlobContours_ToTransform.size()){
			m_bIsTheSame = false;
			m_vArrayMatchContours.resize(m_vvoBlobContours_ToTransform.size(),-1);
			m_iFrameCountDown = 5;
		}
		
		std::vector<cv::DMatch> voCurrBlobMatches;
#if USE_PER_BLOB_SHAPE_DIST
		//Process array contours, find the corresponding contour between Vis & IR
		for (size_t n = 0; n<m_vvoBlobContours_ToTransform.size() && n<m_vArrayMatchContours.size(); ++n) {
            float fMinBlobSimilarityScore = FLT_MAX;
            std::vector<cv::DMatch> voBestBlobMatches;
            size_t nBestContourIdx = 0;
			size_t m;
			double dTmpRatio;


			//Reduce Search space
			if (m_iFrameCountDown == 0 && m_bIsTheSame){
				m = m_vArrayMatchContours[n];
				if (m >= m_vvoBlobContours.size())
					continue;

				if (m_vvoBlobContours_ToTransform[n].size()>MIN_TPS_KEYPOINTS_COUNT && m_vvoBlobContours[m].size()>MIN_TPS_KEYPOINTS_COUNT)
					m_pShapeDistExtr->setTransformAlgorithm(m_pShapeTransf_TPS);
				else if (m_vvoBlobContours_ToTransform[n].size() > MIN_KEYPOINTS_COUNT && m_vvoBlobContours[m].size() > MIN_KEYPOINTS_COUNT)
					m_pShapeDistExtr->setTransformAlgorithm(m_pShapeTransf_Aff);
				else
					continue;
				//Compute distance between 2 contours
				float fCurrBlobSimilarityScore = m_pShapeDistExtr->computeDistance(m_vvoBlobContours_ToTransform[n], m_vvoBlobContours[m]);

				//find the min distance
				if (fCurrBlobSimilarityScore<fMinBlobSimilarityScore) {
					fMinBlobSimilarityScore = fCurrBlobSimilarityScore;
					voCurrBlobMatches = m_pShapeDistExtr->getLatestMatches();
					voBestBlobMatches.clear();
					for (size_t k = 0; k<voCurrBlobMatches.size(); ++k)
					if (voCurrBlobMatches[k].queryIdx<(int)m_vvoBlobContours_ToTransform[n].size() && voCurrBlobMatches[k].trainIdx<(int)m_vvoBlobContours[m].size())
						voBestBlobMatches.push_back(voCurrBlobMatches[k]);
					nBestContourIdx = m;
				}

			}
			else
			// done
			for(m=0; m<m_vvoBlobContours.size(); ++m) {
				//std::cout << m_bIsTheSame<<" "<<m_vArrayMatchContours.size() << " " << m_vvoBlobContours_ToTransform.size() << std::endl;
				//ok
				if(m_vvoBlobContours_ToTransform[n].size()>MIN_TPS_KEYPOINTS_COUNT && m_vvoBlobContours[m].size()>MIN_TPS_KEYPOINTS_COUNT)
					m_pShapeDistExtr->setTransformAlgorithm(m_pShapeTransf_TPS);
				else if(m_vvoBlobContours_ToTransform[n].size()>MIN_KEYPOINTS_COUNT && m_vvoBlobContours[m].size()>MIN_KEYPOINTS_COUNT)
					m_pShapeDistExtr->setTransformAlgorithm(m_pShapeTransf_Aff);
				else
					continue;
				//done

				//Compute distance between 2 contours
				float fCurrBlobSimilarityScore = m_pShapeDistExtr->computeDistance(m_vvoBlobContours_ToTransform[n], m_vvoBlobContours[m]);

				//find the min distance
				if(fCurrBlobSimilarityScore<fMinBlobSimilarityScore) {
					fMinBlobSimilarityScore = fCurrBlobSimilarityScore;
					voCurrBlobMatches = m_pShapeDistExtr->getLatestMatches();
					voBestBlobMatches.clear();
					for(size_t k=0; k<voCurrBlobMatches.size(); ++k)
						if(voCurrBlobMatches[k].queryIdx<(int)m_vvoBlobContours_ToTransform[n].size() && voCurrBlobMatches[k].trainIdx<(int)m_vvoBlobContours[m].size())
							voBestBlobMatches.push_back(voCurrBlobMatches[k]);
					nBestContourIdx = m;

					m_vArrayMatchContours[n] = m;
				}
			}
			//save nearest disparity contours
			if (fMinBlobSimilarityScore<MIN_SIMILARITY_SCORE && !voBestBlobMatches.empty()) {
				vvoFiltBlobContours_ToTransform.push_back(m_vvoBlobContours_ToTransform[n]);
				vvoFiltBlobContours.push_back(m_vvoBlobContours[nBestContourIdx]);
				vvoFiltBlobKeyPointsMatches.push_back(voBestBlobMatches);
			}  
        }
		//End loop for find corresponding

#else //!USE_PER_BLOB_SHAPE_DIST
        std::vector<cv::Point> voFullBlobContours_ToTransform, voFullBlobContours;
        for(size_t n=0; n<m_vvoBlobContours_ToTransform.size(); ++n)
            voFullBlobContours_ToTransform.insert(voFullBlobContours_ToTransform.end(),m_vvoBlobContours_ToTransform[n].begin(),m_vvoBlobContours_ToTransform[n].end());
        for(size_t m=0; m<m_vvoBlobContours.size(); ++m)
            voFullBlobContours.insert(voFullBlobContours.end(),m_vvoBlobContours[m].begin(),m_vvoBlobContours[m].end());
        if(voFullBlobContours_ToTransform.size()>MIN_TPS_KEYPOINTS_COUNT && voFullBlobContours.size()>MIN_TPS_KEYPOINTS_COUNT) {
            float fCurrBlobSimilarityScore = pShapeDistExtr->computeDistance(voFullBlobContours_ToTransform,voFullBlobContours);
            std::vector<cv::DMatch> voBlobMatches, voCurrBlobMatches = pShapeDistExtr->getLatestMatches();
            for(size_t k=0; k<voCurrBlobMatches.size(); ++k)
                if(voCurrBlobMatches[k].queryIdx<(int)voFullBlobContours_ToTransform.size() && voCurrBlobMatches[k].trainIdx<(int)voFullBlobContours.size())
                    voBlobMatches.push_back(voCurrBlobMatches[k]);
            if(fCurrBlobSimilarityScore<MIN_SIMILARITY_SCORE && !voBlobMatches.empty()) {
                vvoFiltBlobContours_ToTransform.push_back(voFullBlobContours_ToTransform);
                vvoFiltBlobContours.push_back(voFullBlobContours);
                vvoFiltBlobKeyPointsMatches.push_back(voBlobMatches);
            }
        }
#endif //!USE_PER_BLOB_SHAPE_DIST
        if(!vvoFiltBlobContours_ToTransform.empty() && !vvoFiltBlobContours.empty()) {
            CV_Assert(vvoFiltBlobContours_ToTransform.size()==vvoFiltBlobContours.size());
            size_t nCurrKeyPointsCount=0;

            for(size_t n=0; n<vvoFiltBlobContours_ToTransform.size(); ++n)
                nCurrKeyPointsCount += vvoFiltBlobKeyPointsMatches[n].size();

			//Guarantee exist corresponding
            if(nCurrKeyPointsCount>0) {
                m_vvoRecentFiltKeyPoints_ToTransform[m_nCurrFrameStackIdx].resize(nCurrKeyPointsCount);
                m_vvoRecentFiltKeyPoints[m_nCurrFrameStackIdx].resize(nCurrKeyPointsCount);

				//Nhet tat ca cac contour vao cung 1 vector 2 chieu. m_nCurrFrameStackIdx la the hien dang o frame thu may trong video
                for(size_t n=0,nKeyPointIdx=0; n<vvoFiltBlobContours_ToTransform.size(); ++n) {
                    for(size_t m=0; m<vvoFiltBlobKeyPointsMatches[n].size(); ++m) {
                        m_vvoRecentFiltKeyPoints_ToTransform[m_nCurrFrameStackIdx][nKeyPointIdx] = vvoFiltBlobContours_ToTransform[n][vvoFiltBlobKeyPointsMatches[n][m].queryIdx];
                        m_vvoRecentFiltKeyPoints[m_nCurrFrameStackIdx][nKeyPointIdx] = vvoFiltBlobContours[n][vvoFiltBlobKeyPointsMatches[n][m].trainIdx];
                        ++nKeyPointIdx;
                    }
                }

				//circle??
                const size_t nCurrModelCoordsCount = (m_nCurrModelLastCoordIdx<m_nCurrModelNextCoordIdx)?(m_nCurrModelNextCoordIdx-m_nCurrModelLastCoordIdx):(m_vfModelCoords_ToTransform.size()-m_nCurrModelLastCoordIdx+m_nCurrModelNextCoordIdx);

				//guarantee not over flow?
                if(nCurrModelCoordsCount+nCurrKeyPointsCount*2>m_vfModelCoords_ToTransform.size()) {
                    const size_t nOldMaxModelCoordsCount = m_vfModelCoords_ToTransform.size();
                    const size_t nNewMaxModelCoordsCount = (nOldMaxModelCoordsCount+nCurrKeyPointsCount*2)*2;
                    ReallocModelVector(m_vfModelCoords_ToTransform,nNewMaxModelCoordsCount,m_nCurrModelNextCoordIdx,m_nCurrModelLastCoordIdx);
                    ReallocModelVector(m_vfModelCoords,nNewMaxModelCoordsCount,m_nCurrModelNextCoordIdx,m_nCurrModelLastCoordIdx);
                    ReallocModelVector(m_vnModelCoordsPersistence,nNewMaxModelCoordsCount/2,m_nCurrModelNextCoordIdx/2,m_nCurrModelLastCoordIdx/2);
                    m_nCurrModelNextCoordIdx = nCurrModelCoordsCount;
                    m_nCurrModelLastCoordIdx = 0;
                }

                float* pfCoords_ToTransform = m_vfModelCoords_ToTransform.data();
                float* pfCoords = m_vfModelCoords.data();
                int* pnPersistence = m_vnModelCoordsPersistence.data();
                const size_t nMaxModelCoordsCount = m_vfModelCoords_ToTransform.size();

                if(!m_bLoopedAtLeastOnce) {
                    for(size_t n=0; n<nCurrKeyPointsCount; ++n) {
                        pnPersistence[m_nCurrModelNextCoordIdx/2] = 0;
                        pfCoords_ToTransform[m_nCurrModelNextCoordIdx] = (float)m_vvoRecentFiltKeyPoints_ToTransform[m_nCurrFrameStackIdx][n].x;
                        pfCoords[m_nCurrModelNextCoordIdx] = (float)m_vvoRecentFiltKeyPoints[m_nCurrFrameStackIdx][n].x;
                        ++m_nCurrModelNextCoordIdx;

                        pfCoords_ToTransform[m_nCurrModelNextCoordIdx] = (float)m_vvoRecentFiltKeyPoints_ToTransform[m_nCurrFrameStackIdx][n].y;
                        pfCoords[m_nCurrModelNextCoordIdx] = (float)m_vvoRecentFiltKeyPoints[m_nCurrFrameStackIdx][n].y;
                        ++m_nCurrModelNextCoordIdx %= nMaxModelCoordsCount;
                    }
                }
                else {
                    if(m_nCurrModelLastCoordIdx>=m_nCurrModelNextCoordIdx) {
                        ReallocModelVector(m_vfModelCoords_ToTransform,m_vfModelCoords_ToTransform.size(),m_nCurrModelNextCoordIdx,m_nCurrModelLastCoordIdx);
                        pfCoords_ToTransform = m_vfModelCoords_ToTransform.data();

                        ReallocModelVector(m_vfModelCoords,m_vfModelCoords.size(),m_nCurrModelNextCoordIdx,m_nCurrModelLastCoordIdx);
                        pfCoords = m_vfModelCoords.data();

                        ReallocModelVector(m_vnModelCoordsPersistence,m_vnModelCoordsPersistence.size(),m_nCurrModelNextCoordIdx/2,m_nCurrModelLastCoordIdx/2);
                        pnPersistence = m_vnModelCoordsPersistence.data();

                        m_nCurrModelNextCoordIdx = nCurrModelCoordsCount;
                        m_nCurrModelLastCoordIdx = 0;
                    }
                    
					//Compute outlier reservoir
					//what???
					cv::Mat oPersistenceMap(1,m_nCurrModelNextCoordIdx/2,CV_32SC1,pnPersistence);
                    double dMinPersistenceVal,dMaxPersistenceVal;

                    cv::minMaxIdx(oPersistenceMap,&dMinPersistenceVal,&dMaxPersistenceVal);
                    if(dMinPersistenceVal>0)
                        oPersistenceMap -= (int)dMinPersistenceVal;

                    for(size_t n=0; n<nCurrKeyPointsCount; ++n) {
                        const size_t nRandModelCoordIdx = m_nCurrModelLastCoordIdx+((rand()*2)%m_nCurrModelNextCoordIdx);
                        if(pnPersistence[nRandModelCoordIdx/2]<=0) {
                            pfCoords_ToTransform[nRandModelCoordIdx] = (float)m_vvoRecentFiltKeyPoints_ToTransform[m_nCurrFrameStackIdx][n].x;
                            pfCoords[nRandModelCoordIdx] = (float)m_vvoRecentFiltKeyPoints[m_nCurrFrameStackIdx][n].x;
                            pfCoords_ToTransform[nRandModelCoordIdx+1] = (float)m_vvoRecentFiltKeyPoints_ToTransform[m_nCurrFrameStackIdx][n].y;
                            pfCoords[nRandModelCoordIdx+1] = (float)m_vvoRecentFiltKeyPoints[m_nCurrFrameStackIdx][n].y;
                        }
                    }

                }

                CV_Assert((m_nCurrModelNextCoordIdx%2)==0 && (m_nCurrModelLastCoordIdx%2)==0);
                const size_t nNewModelCoordsCount = (m_nCurrModelLastCoordIdx<m_nCurrModelNextCoordIdx)?(m_nCurrModelNextCoordIdx-m_nCurrModelLastCoordIdx):(m_vfModelCoords_ToTransform.size()-m_nCurrModelLastCoordIdx+m_nCurrModelNextCoordIdx);

                if(nNewModelCoordsCount>8) { // need at least 8 correspondances to determine a general homography transformation (ignoring scaling factor)
                    if(m_nCurrModelLastCoordIdx<m_nCurrModelNextCoordIdx) {
                        m_oKeyPoints_ToTransform = cv::Mat((m_nCurrModelNextCoordIdx-m_nCurrModelLastCoordIdx)/2,1,CV_32FC2,pfCoords_ToTransform+m_nCurrModelLastCoordIdx);
                        m_oKeyPoints = cv::Mat((m_nCurrModelNextCoordIdx-m_nCurrModelLastCoordIdx)/2,1,CV_32FC2,pfCoords+m_nCurrModelLastCoordIdx);
                    }
                    else {
                        CV_Assert(!m_bLoopedAtLeastOnce);
                        m_oKeyPoints_ToTransform.create((nMaxModelCoordsCount-m_nCurrModelLastCoordIdx+m_nCurrModelNextCoordIdx)/2,1,CV_32FC2);
                        m_oKeyPoints.create((nMaxModelCoordsCount-m_nCurrModelLastCoordIdx+m_nCurrModelNextCoordIdx)/2,1,CV_32FC2);

                        const size_t nFirstBlockLength = nMaxModelCoordsCount-m_nCurrModelLastCoordIdx;
                        memcpy(m_oKeyPoints_ToTransform.data,pfCoords_ToTransform+m_nCurrModelLastCoordIdx,sizeof(float)*nFirstBlockLength);
                        memcpy(m_oKeyPoints.data,pfCoords+m_nCurrModelLastCoordIdx,sizeof(float)*nFirstBlockLength);

                        if(m_nCurrModelNextCoordIdx>0) {
                            const size_t nSecondBlockLength = m_nCurrModelNextCoordIdx;
                            memcpy(((float*)m_oKeyPoints_ToTransform.data)+nFirstBlockLength,pfCoords_ToTransform,sizeof(float)*nSecondBlockLength);
                            memcpy(((float*)m_oKeyPoints.data)+nFirstBlockLength,pfCoords,sizeof(float)*nSecondBlockLength);
                        }
                    }
                    
					//HOT: begin HERE!!

					cv::Mat oOutliersMask;
                    m_oCurrTransMat_inv = cv::findHomography(m_oKeyPoints_ToTransform,m_oKeyPoints,oOutliersMask,cv::RANSAC);
				
					//Tinh Homography
					CalculateHomography(oForeground_ToTransform, oForeground, m_oCurrTransMat_inv);

					//Tinh Fundamental
					if (isCalculatingFundamental == true)
						drawEpipolarLines(oForeground_ToTransform, oForeground, m_oCurrTransMat_inv);
					
						

                    if(!m_oCurrTransMat_inv.empty()) {
                        cv::invert(m_oCurrTransMat_inv,m_oCurrTransMat);

                        m_oLatestTransMat = m_oCurrTransMat.clone();
                        m_oLatestTransMat_inv = m_oCurrTransMat_inv.clone();

                        for(size_t n=m_nCurrModelLastCoordIdx, nLocalIdx=0; n!=m_nCurrModelNextCoordIdx; (n+=2)%=nMaxModelCoordsCount)
                            if(oOutliersMask.at<uchar>(nLocalIdx++))
                                ++pnPersistence[n/2];
                            else
                                --pnPersistence[n/2];
                    }
                }
                ++m_nCurrFrameStackIdx %= NB_FRAME_MEMORY;
                if(!m_bLoopedAtLeastOnce && m_nCurrFrameStackIdx==0)
                    m_bLoopedAtLeastOnce = true;
                if(!m_bLoopedAtLeastOnce)
                    (m_nCurrModelLastCoordIdx+=m_vvoRecentFiltKeyPoints_ToTransform[m_nCurrFrameStackIdx].size()*2) %= nMaxModelCoordsCount;
            }
        }
    }

	//Ket thuc xu li cai j do, cai nay la sau if findcontours

    if(!m_oLatestTransMat.empty()) {
        const cv::Size oTransformedImageSize = oForeground.size();
        m_oLatestTransformedForeground.create(oTransformedImageSize,CV_8UC1);
        m_oBestTransformedForeground.create(oTransformedImageSize,CV_8UC1);



		//Tien hanh transform dua vao ma tran
        cv::warpPerspective(oForeground_ToTransform,m_oLatestTransformedForeground,m_oLatestTransMat,oTransformedImageSize,cv::INTER_NEAREST|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
		cv::warpPerspective(oForeground_ToTransform, m_oBestTransformedForeground, m_oBestTransMat, oTransformedImageSize, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);



        const int nForegroundPxCount = cv::countNonZero(oForeground);
        const int nForegroundPxCount_ToTransform = cv::countNonZero(oForeground_ToTransform);
        const float fLatestForegroundOverlapError = DatasetUtils::CalcForegroundOverlapError(oForeground,m_oLatestTransformedForeground);
        const float fBestForegroundOverlapError = DatasetUtils::CalcForegroundOverlapError(oForeground,m_oBestTransformedForeground);

		//CHU Y CHO NAY!!

		//Homography smoothing
        if(fLatestForegroundOverlapError<fBestForegroundOverlapError &&
                //(float)cv::countNonZero(m_oLatestTransformedForeground)/((!INVERT_FG_PROPORTION)?nForegroundPxCount:nForegroundPxCount_ToTransform)>(float)nForegroundPxCount/nForegroundPxCount_ToTransform &&
                nForegroundPxCount>MIN_BLOB_AREA && nForegroundPxCount_ToTransform>MIN_BLOB_AREA) {
            if(m_fPrevBestForegroundOverlapError>fLatestForegroundOverlapError || fBestForegroundOverlapError>2*fLatestForegroundOverlapError)
                m_nSmoothingCount = DEFAULT_SMOOTH_COUNT;
            else
                ++m_nSmoothingCount;
            const float fSmoothingFactor = ((float)m_nSmoothingCount-1)/m_nSmoothingCount;
            m_fPrevBestForegroundOverlapError = m_fPrevBestForegroundOverlapError*fSmoothingFactor+fLatestForegroundOverlapError*(1-fSmoothingFactor);
			m_oBestTransMat = m_oBestTransMat*fSmoothingFactor + m_oLatestTransMat*(1 - fSmoothingFactor);
			m_oBestTransMat_inv = m_oBestTransMat_inv*fSmoothingFactor + m_oLatestTransMat_inv*(1 - fSmoothingFactor);

        }

		//compare
		compareFundametalAndHomography(isCalculatingFundamental,oForeground_ToTransform, oForeground);

		if (isUsingFundametalMatrix == false)
			cv::warpPerspective(oForeground_ToTransform, m_oBestTransformedForeground, m_oBestChooseTransMat_inv, oForeground_ToTransform.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
		else
			m_oBestTransformedForeground = registrationUsingFundamentalMatrix(oForeground_ToTransform);

		float tmpEr = DatasetUtils::CalcForegroundOverlapError(oForeground, m_oBestTransformedForeground);

		pair<Mat, Mat> tmp_Foreground(oForeground_ToTransform.clone(), oForeground.clone());
		pair<pair<Mat, Mat>, float> tmpStruct(tmp_Foreground, tmpEr);
		m_queueFrames.push(tmpStruct);


    }

}

void MultimodalVideoRegistrAlg::PaintFGRegions(const std::vector<std::vector<cv::Point> >& voPoints, cv::Scalar oPointColor, cv::Scalar oEdgeColor, cv::Mat& oOutput) {
    CV_Assert(!oOutput.empty());
    oOutput = cv::Scalar_<uchar>(0,0,0);
    for(size_t k=0; k<voPoints.size(); ++k) {
        for(size_t l=0; l<voPoints[k].size(); ++l) {
            if(!l)
                cv::circle(oOutput,voPoints[k][l],2,cv::Scalar_<uchar>(255,0,0));
            else
                cv::circle(oOutput,voPoints[k][l],2,oPointColor);
            if(l<voPoints[k].size()-1)
                cv::line(oOutput,voPoints[k][l],voPoints[k][l+1],oEdgeColor);
            else
                cv::line(oOutput,voPoints[k][l],voPoints[k][0],oEdgeColor);
        }
    }
}

void MultimodalVideoRegistrAlg::drawEpipolarLines(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground, cv::Mat &m_oCurrTransMat_inv){
	
	Mat Luan_FundamentalMat = cv::findFundamentalMat(m_oKeyPoints_ToTransform, m_oKeyPoints, CV_FM_RANSAC);
	
	if (isRunSecond == true)
		Luan_FundamentalMat = m_oBestFundamentalMatrix.clone();

	if (Luan_FundamentalMat.cols != 3 || Luan_FundamentalMat.rows != 3)
		return;

	cv::Mat H1, H2;

	cv::stereoRectifyUncalibrated(m_oKeyPoints_ToTransform, m_oKeyPoints,
		Luan_FundamentalMat, oForeground_ToTransform.size(), H1, H2);
	if (isRunSecond == true)
	{
		H1 = H1_Best.clone();
		H2 = H2_Best.clone();
	}

	//Visualize bien doi cua H1, H2
	Mat tmp1 = oForeground_ToTransform.clone();
	cv::warpPerspective(oForeground_ToTransform, tmp1, H1, oForeground_ToTransform.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	Mat tmp2 = oForeground.clone();
	cv::warpPerspective(oForeground, tmp2, H2, oForeground.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	float fCurFundamentalMatrixOverlapError = DatasetUtils::CalcForegroundOverlapError(tmp2, tmp1);

	//imshow("Bien doi cua oForeground_ToTransform", tmp1);
	//imshow("Bien doi cua oForeground", tmp2);

	if (m_oBestFundamentalMatrix.cols == 0)
		m_oBestFundamentalMatrix = Luan_FundamentalMat.clone();

	//Tinh cai m_oBestFundamentalMatrix
	cv::stereoRectifyUncalibrated(m_oKeyPoints_ToTransform, m_oKeyPoints,
		m_oBestFundamentalMatrix, oForeground_ToTransform.size(), H1_Best, H2_Best);

	Mat tmp1_Best = oForeground_ToTransform.clone();
	cv::warpPerspective(oForeground_ToTransform, tmp1_Best, H1_Best, oForeground_ToTransform.size(), cv::INTER_NEAREST , cv::BORDER_CONSTANT);

	Mat tmp2_Best = oForeground.clone();
	cv::warpPerspective(oForeground, tmp2_Best, H2_Best, oForeground.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	fCurFundamentalMatrixOverlapError_Best = DatasetUtils::CalcForegroundOverlapError(tmp2_Best, tmp1_Best);

	if (fCurFundamentalMatrixOverlapError < fCurFundamentalMatrixOverlapError_Best){
		fCurFundamentalMatrixOverlapError_Best = fCurFundamentalMatrixOverlapError;
		//cout << "error = "<<fCurFundamentalMatrixOverlapError_Best << endl;
		m_oBestFundamentalMatrix = Luan_FundamentalMat.clone();
		tmp1_Best = tmp1.clone();
		tmp2_Best = tmp2.clone();
		H1_Best = H1.clone();
		H2_Best = H2.clone();
	}

	
	//imshow("Bien doi cua oForeground_ToTransform_Best", tmp1_Best);
	//imshow("Bien doi cua oForeground_Best", tmp2_Best);

	Mat luan_tmp1_Best = tmp1_Best.clone();
	Mat luan_tmp2_Best = tmp2_Best.clone();

	//Moi them de tinh shape =))
	cv::Ptr<cv::ShapeContextDistanceExtractor> ShapeDistExtrH1H2 = cv::createShapeContextDistanceExtractor(15, 5, 0.1f, 4.0f);
	std::vector<std::vector<cv::Point> > contourH1;
	std::vector<std::vector<cv::Point> > contourH2;
	std::vector<int> vTransValue;
	int finalTransValue = 0;

	cv::findContours(tmp1_Best, contourH1, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	cv::findContours(tmp2_Best, contourH2, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

	std::vector<std::vector<cv::Point> > vvoFiltBlobContoursH1, vvoFiltBlobContoursH2;
	std::vector<std::vector<cv::DMatch> > vvoFiltBlobKeyPointsMatchesH1H2;

	if (!contourH1.empty() && !contourH2.empty()) {

		//cout << "contour = " << contourH1.size() << " " << contourH2.size() << endl;
		for (size_t n = 0; n<contourH1.size(); ++n) {
			float fMinBlobSimilarityScore = FLT_MAX;
			std::vector<cv::DMatch> voBestBlobMatches;
			size_t nBestContourIdx = 0;

			//Moi them
			if (contourH1[n].size() < 3)
				continue;

			for (size_t m = 0; m<contourH2.size(); ++m) {
				//Moi them
				if (contourH2[m].size() < 3)
					continue;

				if (contourH1[n].size()>MIN_TPS_KEYPOINTS_COUNT && contourH2[m].size()>MIN_TPS_KEYPOINTS_COUNT)
					ShapeDistExtrH1H2->setTransformAlgorithm(m_pShapeTransf_TPS);
				else if (contourH1[n].size() > MIN_KEYPOINTS_COUNT && contourH2[m].size() > MIN_KEYPOINTS_COUNT)
					ShapeDistExtrH1H2->setTransformAlgorithm(m_pShapeTransf_Aff);
				else
					continue;

				float fCurrBlobSimilarityScore = ShapeDistExtrH1H2->computeDistance(contourH1[n], contourH2[m]);

				if (fCurrBlobSimilarityScore < fMinBlobSimilarityScore) {
					fMinBlobSimilarityScore = fCurrBlobSimilarityScore;
					std::vector<cv::DMatch> voCurrBlobMatches = ShapeDistExtrH1H2->getLatestMatches();
					voBestBlobMatches.clear();
					for (size_t k = 0; k < voCurrBlobMatches.size(); ++k)
					if (voCurrBlobMatches[k].queryIdx < (int)contourH1[n].size() && voCurrBlobMatches[k].trainIdx < (int)contourH2[m].size())
						voBestBlobMatches.push_back(voCurrBlobMatches[k]);
					nBestContourIdx = m;
				}
			}
			if (fMinBlobSimilarityScore < MIN_SIMILARITY_SCORE && !voBestBlobMatches.empty()) {
				vvoFiltBlobContoursH1.push_back(contourH1[n]);
				vvoFiltBlobContoursH2.push_back(contourH2[nBestContourIdx]);
				vvoFiltBlobKeyPointsMatchesH1H2.push_back(voBestBlobMatches);
			}
		}

		//Tinh gia tri translate
		vTransValue.resize(vvoFiltBlobKeyPointsMatchesH1H2.size());
		for (int i = 0; i < vTransValue.size(); i++){
			int idxX = 0;
			float minDistance = INT_MAX;
			for (int j = 0; j < vvoFiltBlobKeyPointsMatchesH1H2[i].size(); j++){
				if (vvoFiltBlobKeyPointsMatchesH1H2[i][j].distance < minDistance){
					minDistance = vvoFiltBlobKeyPointsMatchesH1H2[i][j].distance;
					idxX = j;
				}
			}

			//vTransValue[i] = vvoFiltBlobContoursH1[i][vvoFiltBlobKeyPointsMatchesH1H2[i][idxX].queryIdx].x - vvoFiltBlobContoursH2[i][vvoFiltBlobKeyPointsMatchesH1H2[i][idxX].trainIdx].x;

			int tmpDis = 0;
			for (int j = 0; j < vvoFiltBlobKeyPointsMatchesH1H2[i].size(); j++){
				tmpDis += vvoFiltBlobContoursH1[i][vvoFiltBlobKeyPointsMatchesH1H2[i][j].queryIdx].x - vvoFiltBlobContoursH2[i][vvoFiltBlobKeyPointsMatchesH1H2[i][j].trainIdx].x;
			}
			
			int sizeQuery = vvoFiltBlobKeyPointsMatchesH1H2[i].size();
			vTransValue[i] = tmpDis / sizeQuery;
			
			//cout <<vTransValue[i] << " ";
			finalTransValue += vTransValue[i];
		}

		//cout << " vTransValue.size()=" << vTransValue.size() << " ";
		//Cai final nay chi thich hop cho planar
		finalTransValue = vTransValue.size()>0 ? finalTransValue / vTransValue.size() : finalTransValue;

		//cout << "Match: ===== " << vvoFiltBlobKeyPointsMatchesH1H2.size() << endl;
		//for (int i = 0; i < vvoFiltBlobKeyPointsMatchesH1H2.size(); i++)
		//cout << ShapeDistExtrH1H2->computeDistance(vvoFiltBlobContoursH1[i], vvoFiltBlobContoursH2[i])<<" ";

		//cout << " ok" << endl;

	}

	bestTransTMatrix = cv::Mat::eye(3, 3, CV_64FC1);
	bestTransTMatrix.at<double>(0, 2) = finalTransValue;

	

	Mat bestTransFundMatrix = H1_Best*bestTransTMatrix*H2_Best.inv();
		
	Mat translateH1toH2(oForeground_ToTransform.size() * 3, oForeground_ToTransform.type());
	Mat tamH1toH2 = luan_tmp1_Best.clone();

	//Da chuyen vo trong ham
	//cv::warpPerspective(oForeground_ToTransform, translateH1toH2, H1_Best, oForeground_ToTransform.size() * 3, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	//cv::warpPerspective(translateH1toH2, translateH1toH2, bestTransTMatrix, oForeground_ToTransform.size() * 3, cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	//cv::warpPerspective(translateH1toH2, translateH1toH2, H2_Best.inv(), oForeground_ToTransform.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	
	
	translateH1toH2 = registrationUsingFundamentalMatrix(oForeground_ToTransform);

	//imshow("Good", translateH1toH2 / 2 + oForeground / 2);
	
	
	
	cv::warpPerspective(oForeground_ToTransform, tamH1toH2, H1_Best, oForeground_ToTransform.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	
	
	m_AdaptiveFundamentalFrame = Mat::zeros(oForeground_ToTransform.size(), oForeground_ToTransform.type());
	if (vTransValue.size() > 0){
		
		Mat outTam = registrationBlobs(tamH1toH2, luan_tmp2_Best, vvoFiltBlobContoursH1, vTransValue);
		imshow("Reg Before Invert", outTam / 2 + luan_tmp2_Best / 2);

		afterH1 = tamH1toH2.clone();
		beforeH2 = outTam.clone();

		cv::warpPerspective(outTam, outTam, H2_Best.inv(), oForeground_ToTransform.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

		m_AdaptiveFundamentalFrame = outTam.clone();
		imshow("Registration result", outTam / 2 + oForeground / 2);
	}
	
	//Bo comment nay
	//Mat translateH1toH2 = oForeground_ToTransform.clone();
	//cv::warpPerspective(luan_tmp1_Best, translateH1toH2, bestTransTMatrix, oForeground_ToTransform.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	////imshow("Overlap sau khi Trans cua ToTransform_Best", (translateH1toH2) / 2 + luan_tmp2_Best / 2);


	//cv::warpPerspective(translateH1toH2, tmp2, H2_Best.inv(), oForeground_ToTransform.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	// imshow("Sau bien doi thong qua m_oBestFundamentalMatrix", tmp2 / 2 + oForeground / 2);

	 //cv::warpPerspective(oForeground, tmp2, bestTransFundMatrix, oForeground_ToTransform.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	 //imshow("VL", tmp2 / 2 + oForeground / 2);



	 
	//if (fLatestForegroundOverlapError < m_curOverlapError){
	//	m_curOverlapError = fLatestForegroundOverlapError;
	//	m_oBestTransMat = resMat.clone();
	//}
	
		
}


void MultimodalVideoRegistrAlg::FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs)
{
	blobs.clear();

	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground

	cv::Mat label_image;
	binary.convertTo(label_image, CV_32SC1);

	int label_count = 2; // starts at 2 because 0,1 are used already

	for (int y = 0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		for (int x = 0; x < label_image.cols; x++) {
			if (row[x] != 1) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);

			std::vector <cv::Point2i> blob;

			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				int *row2 = (int*)label_image.ptr(i);
				for (int j = rect.x; j < (rect.x + rect.width); j++) {
					if (row2[j] != label_count) {
						continue;
					}

					blob.push_back(cv::Point2i(j, i));
				}
			}

			blobs.push_back(blob);

			label_count++;
		}
	}
}

void MultimodalVideoRegistrAlg::correctForeground(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground){

}

void MultimodalVideoRegistrAlg::setTransMat(const MultimodalVideoRegistrAlg& oAlg2){


	m_oBestChooseTransMat = oAlg2.GetTransformationMatrix(false).clone();
	m_oBestChooseTransMat_inv = oAlg2.GetTransformationMatrix(true).clone();
	m_fPrevBestForegroundOverlapError = oAlg2.m_fPrevBestForegroundOverlapError;
	m_oLatestTransMat = oAlg2.GetLatestTransMat(false).clone();
	m_oLatestTransMat_inv = oAlg2.GetLatestTransMat(true).clone();
	m_oBestTransMat = oAlg2.GetBestTransMat(false).clone();
	m_oBestTransMat_inv = oAlg2.GetBestTransMat(true).clone();
	m_oCurrTransMat = oAlg2.GetCurrTransMat(false).clone();
	m_oCurrTransMat_inv = oAlg2.GetCurrTransMat(true).clone();
	m_nSmoothingCount = oAlg2.m_nSmoothingCount;
}

bool myfunction(vector<Point> i, vector<Point> j) { return (i.size()>j.size()); }

Mat MultimodalVideoRegistrAlg::registrationBlobs(cv::Mat inputMat, cv::Mat refMat, std::vector<std::vector<cv::Point> > listContour, std::vector<int> vValueTrans){
	//Input la anh da rectify
	//Output la anh truoc khi nhan H2
	std::vector < std::vector<cv::Point2i > > blobs;
	cv::Mat input_binary;
	cv::Mat outputMat = Mat::zeros(inputMat.size(), inputMat.type());

	Mat tmp = inputMat.clone();
	cv::threshold(inputMat, input_binary, 0.0, 1.0, cv::THRESH_BINARY);
	FindBlobs(input_binary, blobs);
	std::sort(blobs.begin(), blobs.end(), myfunction);

	
	//imshow("inputMat", inputMat);

	blobs.resize(vValueTrans.size());

	for (size_t i = 0; i < blobs.size(); i++) {
		Mat pic = Mat::zeros(inputMat.size(), inputMat.type());
		//Mat pic2 = Mat::zeros(inputMat.size(), inputMat.type());
		//cout << "blobs size= " << blobs[i].size() << endl;

		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;
			pic.at<uchar>(y, x) = 255;
		}
		
		
		int m;
		for (m = 0; m < listContour.size(); m++){
			int n;
			for (n = 0; n < listContour[m].size(); n++){
				if (pic.at<uchar>(listContour[m][n].y, listContour[m][n].x) < 255)
					break;
			}
			if (n > listContour[m].size()/2){
				//cout << "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa " << blobs[i].size()<<endl;
				break;
			}
				
		}

		if (m == listContour.size())
			continue;
		
		int t = 0;
		float minRef = 1, prevRef=1;
		float pointRight = 1, pointLeft = 1, pointCur = 1;
		Mat outputMat2 = outputMat.clone();
		Mat outputMat1 = outputMat.clone();

		while (1){
			if (t > inputMat.cols)
				break;
			Mat tmpImage = outputMat.clone();
			
			for (size_t j = 0; j < blobs[i].size(); j++) {
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;

				if (x + vValueTrans[m] + t < outputMat.cols && x + vValueTrans[m] + t >= 0)
					tmpImage.at<uchar>(y, x + vValueTrans[m] + t) = 255;
			}
			float errorTmp = DatasetUtils::CalcForegroundOverlapError(tmpImage, refMat);
			if (errorTmp > prevRef)
				break;
			else
				outputMat1 = tmpImage.clone();
			t++;
			prevRef = errorTmp;
		}
		t--;
		pointRight = prevRef;
		prevRef = 1;
		int disRight = vValueTrans[m] + t;
		//cout << "dis = " << vValueTrans[m] + t << " ";
		t = 0;
		
		while (1){
			if (t < -inputMat.cols)
				break;
			Mat tmpImage = outputMat.clone();

			for (size_t j = 0; j < blobs[i].size(); j++) {
				int x = blobs[i][j].x;
				int y = blobs[i][j].y;

				if (x + vValueTrans[m] + t < outputMat.cols && x + vValueTrans[m] + t >= 0)
					tmpImage.at<uchar>(y, x + vValueTrans[m] + t) = 255;
			}
			float errorTmp = DatasetUtils::CalcForegroundOverlapError(tmpImage, refMat);
			if (errorTmp > prevRef)
				break;
			else
				outputMat2 = tmpImage.clone();
			t--;
			prevRef = errorTmp;
		}
		t++;
		//cout << "dis = " << vValueTrans[m] + t << " ";
		pointLeft = prevRef;
		int disLeft = vValueTrans[m] + t;
		disparityFinal.push_back(disRight);
		disparityFinal.push_back(disLeft);

		if (pointLeft < pointRight){
			outputMat = outputMat2.clone();
		}
		else{
			outputMat = outputMat1.clone();
		}
	}

	

	if (vValueTrans.size() == 0)
		outputMat = inputMat;

	return outputMat.clone();
}


void MultimodalVideoRegistrAlg::PreProcessForeground(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground) {
	//cout << "size=" <<m_List.size() << endl;

	std::vector < std::vector<cv::Point2i > > blobs, blobs2;
	cv::Mat oForeground_ToTransform_binary, oForeground_binary;

	Mat tmp = oForeground.clone();
	Mat tmp2 = oForeground.clone();


	cv::warpPerspective(oForeground_ToTransform, tmp2, m_oBestChooseTransMat, oForeground_ToTransform.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);

	cv::threshold(oForeground, oForeground_binary, 0.0, 1.0, cv::THRESH_BINARY);
	FindBlobs(oForeground_binary, blobs);

	imshow("oForeground_ToTransform0000000", tmp2 / 2 + oForeground/2);

	for (size_t i = 0; i < blobs.size(); i++) {
		bool isHave = false;
		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			for (int m = -5; m <= 5; m++)
			for (int n = -5; n <= 5; n++)
			if (y + m >= 0 && y + m <tmp.rows && x + n >= 0 && x + n <tmp.cols && tmp2.at<uchar>(y + m, x + n) == 255)
				isHave = true;
				
		}
		if (isHave == false)
		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			oForeground.at<uchar>(y, x) = 0;
		}
	}

	cv::warpPerspective(oForeground, tmp, m_oBestChooseTransMat_inv, oForeground.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);

	cv::threshold(oForeground_ToTransform, oForeground_ToTransform_binary, 0.0, 1.0, cv::THRESH_BINARY);
	FindBlobs(oForeground_ToTransform_binary, blobs2);

	//imshow("oForeground0", oForeground);
	for (size_t i = 0; i < blobs2.size(); i++) {
		bool isHave = false;
		for (size_t j = 0; j < blobs2[i].size(); j++) {
			int x = blobs2[i][j].x;
			int y = blobs2[i][j].y;

			for (int m = -5; m <= 5; m++)
				for (int n = -5; n <= 5; n++)
				if (y + m >= 0 && y + m <tmp.rows && x + n >= 0 && x + n <tmp.cols && tmp.at<uchar>(y + m, x + n) == 255)
					isHave = true;

		}
		if (isHave == false)
		for (size_t j = 0; j < blobs2[i].size(); j++) {
			int x = blobs2[i][j].x;
			int y = blobs2[i][j].y;

			oForeground_ToTransform.at<uchar>(y, x) = 0;
		}
	}


}


void MultimodalVideoRegistrAlg::CalculateHomography(const cv::Mat& oForeground_ToTransform, const cv::Mat& oForeground, const cv::Mat& TransMat){
	Mat tmp = oForeground_ToTransform.clone();

	if (m_oCurrTransMat_inv.cols != 3 || m_oCurrTransMat_inv.rows != 3)
		return;

	Mat m_oCurrTransMat;
	m_oCurrTransMat = m_oCurrTransMat_inv.inv();
	cv::warpPerspective(oForeground_ToTransform, tmp, m_oCurrTransMat_inv, oForeground_ToTransform.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
	float fLatestForegroundOverlapError = DatasetUtils::CalcForegroundOverlapError(oForeground, tmp);
	if (fLatestForegroundOverlapError < m_curOverlapError){
		m_curOverlapError = fLatestForegroundOverlapError;
		
		//Cai nay quyet dinh cao hay thap ne
		//m_oBestTransMat = m_oCurrTransMat.clone();
		//m_oBestTransMat_inv = m_oCurrTransMat_inv.clone();
		m_oLatestTransMat = m_oCurrTransMat.clone();
		m_oLatestTransMat_inv = m_oCurrTransMat_inv.clone();
	}
	//cout << "m_curOverlapError = " << m_curOverlapError << endl;
}

void MultimodalVideoRegistrAlg::compareFundametalAndHomography(bool isCalculatingFundamental, cv::Mat& oForeground_ToTransform, cv::Mat& oForeground){
	//cach chon cac frame reference de kiem tra xem matrix moi co tot hon k
	if (iCountFrame < 55)
		iCountFrame++;

	if (m_queueFrames.size() >= iCountFrame && iCountFrame > 3){
		float isBad = 0, isBadFundamental = 0, isBadChooseHomo = 0; //TICH LUY SAI SO KHI BIEN DOI, XEM THU MATRIX HIEN TAI CO TOT HON SO VOI TRC K
		int iCount = m_queueFrames.size();
		queue<float> queError, queErrorFundamental;
		while (iCount-- > 0){
			pair<pair<Mat, Mat>, float> tmpStruct = m_queueFrames.front();
			m_queueFrames.pop();
			m_queueFrames.push(tmpStruct);
			Mat tmpMat;

			cv::warpPerspective(tmpStruct.first.first, tmpMat, m_oBestTransMat_inv, tmpStruct.first.first.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
			float tmpForegroundOverlapError = DatasetUtils::CalcForegroundOverlapError(tmpStruct.first.second, tmpMat);
			//cout << tmpForegroundOverlapError << " ";
			//isBad += (tmpForegroundOverlapError - tmpStruct.second);
			isBad += tmpForegroundOverlapError;
			queError.push(tmpForegroundOverlapError);


			cv::warpPerspective(tmpStruct.first.first, tmpMat, m_oBestChooseTransMat, tmpStruct.first.first.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
			float tmpForegroundOverlapErrorChooseHomo = DatasetUtils::CalcForegroundOverlapError(tmpStruct.first.second, tmpMat);
			isBadChooseHomo += tmpForegroundOverlapErrorChooseHomo;


			if (isCalculatingFundamental == true && H1_Best.size().height != 0){
				tmpMat = registrationUsingFundamentalMatrix(tmpStruct.first.first);

				float tmpForegroundOverlapErrorFundamental = DatasetUtils::CalcForegroundOverlapError(tmpStruct.first.second, tmpMat);

				//isBadFundamental += (tmpForegroundOverlapErrorFundamental - tmpStruct.second);
				isBadFundamental += tmpForegroundOverlapErrorFundamental;

				//cout << "Error=" << tmpForegroundOverlapError<<" ErrorFundamental=" << tmpForegroundOverlapErrorFundamental << "tmpStruct.second=" << tmpStruct.second << endl;
				queErrorFundamental.push(tmpForegroundOverlapErrorFundamental);
			}
		}

		float tmpisBad = isBad, tmpisBadFundamental = isBadFundamental;
		//isBad -= minisBad;
		//cout << "minisBad=" << minisBad << " tmpisBad=" << tmpisBad << endl;
		minisBad = min(minisBad, tmpisBad);
		//Dong nay neu comment thi se chi la su dung Fund
		minisBad = min(minisBad, isBadChooseHomo);



		if (isCalculatingFundamental == true && isBadFundamental != 0){
			//	isBadFundamental -= minisBadFundamental;
			//	cout << "minisBadFundamental=" << minisBadFundamental << " tmpisBadFundamental=" << tmpisBadFundamental << endl;
			minisBadFundamental = min(minisBadFundamental, tmpisBadFundamental);
		}
		//cout << "isCalculatingFundamental=" << isCalculatingFundamental<< " isBadFundamental=" << isBadFundamental << " isBad=" << isBad << endl;
		//if ((isCalculatingFundamental == false && isBad > 0) || (isCalculatingFundamental == true && isBadFundamental > 0 && isBad > 0))

		//Chi su dung Homography
		if (isCalculatingFundamental == false){
			if (isBad > minisBad)
			while (iCount-- > 0){
				m_queueFrames.push(m_queueFrames.front());
				m_queueFrames.pop();
			}
			else{
				m_oBestChooseTransMat = m_oBestTransMat.clone();
				m_oBestChooseTransMat_inv = m_oBestTransMat_inv.clone();
				cout << "Thay doi" << endl;
				while (!queError.empty()){
					//cout << queError.front() << endl;
					pair<pair<Mat, Mat>, float> tmpStruct = m_queueFrames.front();
					tmpStruct.second = queError.front();
					m_queueFrames.push(tmpStruct);
					m_queueFrames.pop();
					queError.pop();
				}
			}
		}

		cout << "minisBadFundamental=" << minisBadFundamental << " minisBad=" << minisBad << endl;
		//Su dung Fundamental
		if (isCalculatingFundamental == true){
			if (isBadFundamental > minisBadFundamental && isBad > minisBad)
			while (iCount-- > 0){
				m_queueFrames.push(m_queueFrames.front());
				m_queueFrames.pop();
			}
			else{
				if (isBadFundamental > 0 && isBadFundamental <= minisBadFundamental/* && isBad > minisBad*/){
					if (isBadFundamental <= minisBad){
						isUsingFundametalMatrix = true;

						cout << "isUsingFundametalMatrix----------------------------------" << endl;
						cout << "minisBadFundamental=" << minisBadFundamental << " isBadFundamental=" << isBadFundamental << endl;
						cout << "minisBad=" << minisBad << " isBad=" << isBad << endl;

						H1_Best_Choose = H1_Best.clone();
						H2_Best_Choose = H2_Best.clone();
						bestTransTMatrix_Choose = bestTransTMatrix.clone();

						while (!queErrorFundamental.empty()){
							//cout << queError.front() << endl;
							pair<pair<Mat, Mat>, float> tmpStruct = m_queueFrames.front();
							tmpStruct.second = queErrorFundamental.front();
							m_queueFrames.push(tmpStruct);
							m_queueFrames.pop();
							queErrorFundamental.pop();
						}
					}
				}

				if (/*isBadFundamental > minisBadFundamental && */isBad <= minisBad){
					if (isBad <= minisBadFundamental){
						m_oBestChooseTransMat = m_oBestTransMat.clone();
						m_oBestChooseTransMat_inv = m_oBestTransMat_inv.clone();

						cout << "minisBadFundamental=" << minisBadFundamental << " isBadFundamental=" << isBadFundamental << endl;
						cout << "minisBad=" << minisBad << " isBad=" << isBad << endl;
						isUsingFundametalMatrix = false;

						while (!queError.empty()){

							if (isUsingFundametalMatrix == false){
								pair<pair<Mat, Mat>, float> tmpStruct = m_queueFrames.front();
								tmpStruct.second = queError.front();
								m_queueFrames.push(tmpStruct);
								m_queueFrames.pop();
							}
							queError.pop();
						}
					}
				}


				//if (isBadFundamental < isBad && isBadFundamental < 0 && minisBadFundamental < minisBad){

				//	isUsingFundametalMatrix = true;

				//	cout << "isUsingFundametalMatrix----------------------------------" << endl;

				//	while (!queErrorFundamental.empty()){
				//		//cout << queError.front() << endl;
				//		pair<pair<Mat, Mat>, float> tmpStruct = m_queueFrames.front();
				//		tmpStruct.second = queErrorFundamental.front();
				//		m_queueFrames.push(tmpStruct);
				//		m_queueFrames.pop();
				//		queErrorFundamental.pop();
				//	}
				//}
				//else
				//if (isBad < 0)
				//{
				//	m_oBestChooseTransMat = m_oBestTransMat.clone();
				//	m_oBestChooseTransMat_inv = m_oBestTransMat_inv.clone();

				//	while (!queError.empty()){

				//		if (isUsingFundametalMatrix == false){
				//			pair<pair<Mat, Mat>, float> tmpStruct = m_queueFrames.front();
				//			tmpStruct.second = queError.front();
				//			m_queueFrames.push(tmpStruct);
				//			m_queueFrames.pop();
				//		}
				//		queError.pop();
				//	}
				//	if (isUsingFundametalMatrix == false)
				//		isUsingFundametalMatrix = false;
				//}
			}
		}


		m_queueFrames.pop();

	}
	else{
		//cout << "5 cai dau" << endl;
		m_oBestChooseTransMat = m_oBestTransMat.clone();
		m_oBestChooseTransMat_inv = m_oBestTransMat_inv.clone();
		isUsingFundametalMatrix = false;

	}

	Mat tmpMat;

	cv::warpPerspective(oForeground_ToTransform, tmpMat, m_oBestChooseTransMat, oForeground_ToTransform.size(), cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
	float ForegroundOverlapErrorChooseHomo = DatasetUtils::CalcForegroundOverlapError(oForeground, tmpMat);

	tmpMat = registrationUsingFundamentalMatrix(oForeground_ToTransform);
	float HardForegroundOverlapErrorFundamental = DatasetUtils::CalcForegroundOverlapError(oForeground, tmpMat);

	float AdaptiveForegroundOverlapErrorFundamental = DatasetUtils::CalcForegroundOverlapError(oForeground, m_AdaptiveFundamentalFrame);
	isNewWay = false;
	//if (AdaptiveForegroundOverlapErrorFundamental < min(minisBadFundamental, minisBad))
	//	isNewWay = true;

	if (minisBadFundamental < minisBad)
		isUsingFundametalMatrix = true;
	else
		isUsingFundametalMatrix = false;
}

cv::Mat MultimodalVideoRegistrAlg::registrationUsingFundamentalMatrix(cv::Mat inputMat){
	Mat outputMat(inputMat.size()*3, inputMat.type());
	Mat outMat = inputMat.clone();

	if (H1_Best.cols <= 0)
		return outMat.clone();

	cv::warpPerspective(inputMat, outputMat, H1_Best, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outputMat, bestTransTMatrix, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outMat, H2_Best.inv(), inputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	return outMat.clone();
}

cv::Mat MultimodalVideoRegistrAlg::registrationUsingFundamentalMatrix_inv(cv::Mat inputMat){
	Mat outputMat(inputMat.size()* 3, inputMat.type());
	Mat outMat = inputMat.clone();
	cv::warpPerspective(inputMat, outputMat, H2_Best, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outputMat, bestTransTMatrix.inv(), outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outMat, H1_Best.inv(), inputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	return outMat.clone();
}

cv::Mat MultimodalVideoRegistrAlg::registrationUsingFundamentalMatrix_Choose(cv::Mat inputMat){
	Mat outputMat(inputMat.size() * 3, inputMat.type());
	Mat outMat = inputMat.clone();

	cv::warpPerspective(inputMat, outputMat, H1_Best_Choose, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outputMat, bestTransTMatrix_Choose, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outMat, H2_Best_Choose.inv(), inputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	return outMat.clone();
}

cv::Mat MultimodalVideoRegistrAlg::registrationUsingFundamentalMatrix_inv_Choose(cv::Mat inputMat){
	Mat outputMat(inputMat.size() * 3, inputMat.type());
	Mat outMat = inputMat.clone();
	cv::warpPerspective(inputMat, outputMat, H2_Best_Choose, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outputMat, bestTransTMatrix_Choose.inv(), outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outMat, H1_Best_Choose.inv(), inputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	return outMat.clone();
}

void MultimodalVideoRegistrAlg::AdaptiveFrame(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground, int numFrame){
	if (numFrame == 3){
		oForeground_ToTransform = m_List[iIndexCurrentFrame].first + m_List[2 * iIndexCurrentFrame / 3].first + m_List[iIndexCurrentFrame / 3].first;
		oForeground = m_List[iIndexCurrentFrame].second + m_List[2 * iIndexCurrentFrame / 3].second + m_List[iIndexCurrentFrame / 3].second;
	}
	if (numFrame == 2){
		oForeground_ToTransform = m_List[iIndexCurrentFrame].first+ m_List[iIndexCurrentFrame / 2].first;
		oForeground = m_List[iIndexCurrentFrame].second + m_List[iIndexCurrentFrame / 2].second;
	}
}

Mat MultimodalVideoRegistrAlg::registrationTransform(cv::Mat inputMat, cv::Mat Trans){
	Mat outputMat(inputMat.size() * 3, inputMat.type());
	Mat outMat = inputMat.clone();

	cv::warpPerspective(inputMat, outputMat, H1_Best, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outputMat, Trans, outputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);
	cv::warpPerspective(outputMat, outMat, H2_Best.inv(), inputMat.size(), cv::INTER_NEAREST, cv::BORDER_CONSTANT);

	return outMat.clone();
}