#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/shape.hpp>

#include <vector>
#include <queue>
using namespace std;
using namespace cv;
/*!
 *
 *  MultimodalVideoRegistrAlg v1.0: planar registration only, based on homography estimation
 *
 *  For more details, refer to the 2015 CVPRW (PBVS) paper 'Online Multimodal Video Registration
 *  Based on Shape Matching'. Note that this algorithm relies on OpenCV >= 3.0.0a, patched with
 *  the diff file provided with the rest of the source code.
 *
 */
class MultimodalVideoRegistrAlg {
public:
    //! default constructor, also readies all internal structures for processing
    MultimodalVideoRegistrAlg();
    //! resets all internal structures (i.e. correspondence buffers, reference transformations, smoothing weights)
    void ResetInternalState();
    //! extracts foreground shapes from fg-bg segmentation masks and attempts to register them
	void ProcessForeground(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground, bool isCalculatingFundamental = true);

    //! returns the latest extracted foreground object contours
    inline const std::vector<std::vector<cv::Point> >& GetLatestContours(bool bToTransform) const {return bToTransform?m_vvoBlobContours_ToTransform:m_vvoBlobContours;}
    //! returns the latest estimated homography
    //inline const cv::Mat& GetTransformationMatrix(bool bInvert) const {return bInvert?m_oBestTransMat_inv:m_oBestTransMat;}
	inline const cv::Mat& GetTransformationMatrix(bool bInvert) const { 
		if (isUsingFundametalMatrix == false)
			return bInvert ? m_oBestChooseTransMat_inv : m_oBestChooseTransMat;
		else
			//return bInvert ? m_oBestFundamentalMatrix : m_oBestFundamentalMatrix.inv();
			return bInvert ? m_oBestChooseTransMat : m_oBestChooseTransMat_inv;
	}

    //! utility: paints foreground regions based on contour points lists
    static void PaintFGRegions(const std::vector<std::vector<cv::Point> >& voPoints, cv::Scalar oPointColor, cv::Scalar oEdgeColor, cv::Mat& oOutput);



	//NEW
	void drawEpipolarLines(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground, cv::Mat &curMat);
	void FindBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs);
	void correctForeground(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground);
	void setTransMat(const MultimodalVideoRegistrAlg& oAlg2);
	inline const cv::Mat& GetLatestTransMat(bool bInvert) const { return bInvert ? m_oLatestTransMat_inv : m_oLatestTransMat; }
	inline const cv::Mat& GetBestTransMat(bool bInvert) const { return bInvert ? m_oBestTransMat_inv : m_oBestTransMat; }
	inline const cv::Mat& GetCurrTransMat(bool bInvert) const { return bInvert ? m_oCurrTransMat_inv : m_oCurrTransMat; }

	void PreProcessForeground(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground);
	void CalculateHomography(const cv::Mat& oForeground_ToTransform, const cv::Mat& oForeground, const cv::Mat& TransMat);

	void compareFundametalAndHomography(bool isCalculatingFundamental, cv::Mat& oForeground_ToTransform, cv::Mat& oForeground);
	cv::Mat registrationUsingFundamentalMatrix(cv::Mat inputMat);
	cv::Mat registrationUsingFundamentalMatrix_inv(cv::Mat inputMat);
	cv::Mat registrationUsingFundamentalMatrix_Choose(cv::Mat inputMat);
	cv::Mat registrationUsingFundamentalMatrix_inv_Choose(cv::Mat inputMat);

	void AdaptiveFrame(cv::Mat& oForeground_ToTransform, cv::Mat& oForeground, int numFrame = 1);


	cv::Mat registrationTransform(cv::Mat inputMat, cv::Mat Trans);
	Mat registrationBlobs(cv::Mat inputMat, cv::Mat refMat, std::vector<std::vector<cv::Point> > listContour, std::vector<int> vValueTrans);

	bool isNewWay;

protected:
	void reduceSize(std::vector<std::vector<cv::Point> >& v);
	

private:
    cv::Ptr<cv::ShapeContextDistanceExtractor> m_pShapeDistExtr;
    cv::Ptr<cv::ShapeTransformer> m_pShapeTransf_TPS,m_pShapeTransf_Aff;
    std::vector<std::vector<cv::Point> > m_vvoRecentFiltKeyPoints_ToTransform,m_vvoRecentFiltKeyPoints;
    std::vector<std::vector<cv::Point> > m_vvoBlobContours_ToTransform, m_vvoBlobContours;
    int m_nSmoothingCount;
    bool m_bDebug;
    bool m_bLoopedAtLeastOnce;
    size_t m_nCurrFrameStackIdx;
    size_t m_nCurrModelNextCoordIdx;
    size_t m_nCurrModelLastCoordIdx;
    float m_fPrevBestForegroundOverlapError;
    std::vector<float> m_vfModelCoords_ToTransform,m_vfModelCoords;
    std::vector<int> m_vnModelCoordsPersistence;
    cv::Mat m_oKeyPoints_ToTransform,m_oKeyPoints;
    cv::Mat m_oLatestTransMat,m_oLatestTransMat_inv;
    cv::Mat m_oBestTransMat,m_oBestTransMat_inv;
    cv::Mat m_oCurrTransMat,m_oCurrTransMat_inv;
    cv::Mat m_oLatestTransformedForeground;
    cv::Mat m_oBestTransformedForeground;

	//New one
	std::vector<int> m_vArrayMatchContours;
	bool m_bIsTheSame;
	int m_iFrameCountDown;
	std::vector<std::vector<cv::Point> >* newVec;

	float m_curOverlapError;
	cv::Mat m_oBestChooseTransMat, m_oBestChooseTransMat_inv;
	
	cv::Mat H1_Best, H2_Best, bestTransTMatrix;
	cv::Mat H1_Best_Choose, H2_Best_Choose, bestTransTMatrix_Choose;

	pair<pair<pair<Mat, Mat>, float>, pair<Mat, Mat> > bestFrame;
	float minError;

	vector<pair<Mat, Mat> > m_List;
	

	pair<Mat, Mat> m_referenceFrame;
	
	bool m_isPreProcessed;
	cv::Mat m_oBestFundamentalMatrix;
	cv::Mat bestTransFundMatrix;
	float fCurFundamentalMatrixOverlapError_Best;

	vector<int> m_ArrayDis;
	float m_ratioH1H2;

	float m_ErrorH1H2Cur, m_ErrorH1H2Best;
public:
	float minisBad, minisBadFundamental;
	bool isUsingFundametalMatrix;
	int iCountFrame;
	queue<pair<pair<Mat, Mat>, float> > m_queueFrames;
	int iIndexCurrentFrame;
	Mat m_AdaptiveFundamentalFrame;

	bool isRunSecond;

	Mat afterH1, beforeH2;
	vector<int> disparityFinal;
    template<typename T> inline static void ReallocModelVector(std::vector<T>& vtModel, size_t nNewMaxModelSize, size_t nCurrModelNextIdx, size_t nCurrModelLastIdx) {
        const size_t nOldMaxModelCoordsCount = vtModel.size();
        std::vector<T> vtNewModel(nNewMaxModelSize);
        const size_t nFirstBlockLength = nOldMaxModelCoordsCount-nCurrModelLastIdx;
        memcpy(vtNewModel.data(),vtModel.data()+nCurrModelLastIdx,sizeof(float)*nFirstBlockLength);
        if(nCurrModelNextIdx>0) {
            const size_t nSecondBlockLength = nCurrModelNextIdx;
            memcpy(vtNewModel.data()+nFirstBlockLength,vtModel.data(),sizeof(float)*nSecondBlockLength);
        }
        vtModel = vtNewModel;
    }
};
