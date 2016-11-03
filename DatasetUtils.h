#pragma once

#include <opencv2/opencv.hpp>

namespace DatasetUtils {

    inline float CalcForegroundOverlapError(const cv::Mat& oReferenceFG, const cv::Mat& oCurrFG) {
		//NEW
		if (cv::countNonZero(oReferenceFG | oCurrFG) == 0)
			return 1;
        return 1.0f-(float)(cv::countNonZero(oReferenceFG&oCurrFG))/cv::countNonZero(oReferenceFG|oCurrFG);
    }

    inline cv::Point2d CalcPolyRegError(const cv::Mat& oReferencePts, const cv::Mat& oCurrPts) {
        cv::Point2d oPolyRegError(0,0);
        CV_Assert(!oCurrPts.empty() && oReferencePts.cols==oCurrPts.cols);
        for(int i=0; i<oCurrPts.cols; ++i)
            oPolyRegError += cv::Point2d(abs(oReferencePts.at<double>(0,i)-oCurrPts.at<double>(0,i)/oCurrPts.at<double>(2,i)),abs(oReferencePts.at<double>(1,i)-oCurrPts.at<double>(1,i)/oCurrPts.at<double>(2,i)));
        oPolyRegError/= oCurrPts.cols;
		

        return oPolyRegError;
    }

}; // namespace DatasetUtils
