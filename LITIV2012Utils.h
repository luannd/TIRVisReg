//--------------------------------------------------------
// TIRVisReg
// Copyright (c) 2015 
// Licensed under The GNU License [see LICENSE for details]
// Written by Dinh-Luan Nguyen
// --------------------------------------------------------

#pragma once

#include "DatasetUtils.h"

#ifndef LITIV2012_DATASET_PATH
#error "Need to pre-define the path to the LITIV dataset base folder under 'LITIV2012_DATASET_PATH' before including 'DatasetUtils.h'"
#endif //!defined
#ifndef __LITIV_STR_BASE
#define __LITIV_STR_BASE(x) #x
#define __LITIV_STR(x) __LITIV_STR_BASE(x)
#endif //!defined(__LITIV_STR_BASE)
#ifndef LITIV2012_TEST_SEQUENCE_ID
#error "Need to define the test sequence index under 'LITIV2012_TEST_SEQUENCE_ID' ([1-9])"
#elif (LITIV2012_TEST_SEQUENCE_ID<1 || LITIV2012_TEST_SEQUENCE_ID>9)
#error "Bad test sequence index"
#else // Setup test sequence paths
#ifndef LITIV2012_SEQ_BGS_MASK_FOLDER_NAME
#define LITIV2012_SEQ_BGS_MASK_FOLDER_NAME "segm_pawcs_nobstrap"
#endif //!LITIV2012_SEQ_BGS_MASK_FOLDER_NAME
#define LITIV2012_TEST_SET_PATH               LITIV2012_DATASET_PATH "\\SEQUENCE" __LITIV_STR(LITIV2012_TEST_SEQUENCE_ID) "\\"
#define LITIV2012_TEST_SET_BGS_PATH           LITIV2012_DATASET_PATH LITIV2012_SEQ_BGS_MASK_FOLDER_NAME "\\SEQUENCE" __LITIV_STR(LITIV2012_TEST_SEQUENCE_ID) "\\"
#endif // Setup test sequence paths

namespace DatasetUtils {

    namespace LITIV2012 {

        void ReadTestSeqGroundtruth(cv::Mat& oGTTransMat_THERMAL, cv::Mat& oGTTransMat_VISIBLE, cv::Mat& oPolyListMat_THERMAL, cv::Mat& oPolyListMat_VISIBLE) {
            cv::FileStorage oGTFS(LITIV2012_TEST_SET_PATH "\\GT.yml",cv::FileStorage::READ);
            if(!oGTFS.isOpened())
                CV_Error(-1,"failed to open GT file storage");
            oGTFS["homography"] >> oGTTransMat_THERMAL;
            if(oGTTransMat_THERMAL.empty())
                CV_Error(-1,"failed to read GT trans mat data");
            cv::invert(oGTTransMat_THERMAL,oGTTransMat_VISIBLE);
            oPolyListMat_THERMAL.create(2,3,CV_64FC2);
            oGTFS["thermal_polygons"] >> oPolyListMat_THERMAL;
            if(oPolyListMat_THERMAL.empty())
                CV_Error(-1,"failed to read thermal GT poly mat data");
            oPolyListMat_VISIBLE.create(2,3,CV_64FC2);
            oGTFS["visible_polygons"] >> oPolyListMat_VISIBLE;
            if(oPolyListMat_VISIBLE.empty())
                CV_Error(-1,"failed to read visible GT poly mat data");
        }

        void ConvertPolyPtsMatsToPtsLists(const cv::Mat& oPolyListMat_THERMAL, const cv::Mat& oPolyListMat_VISIBLE, cv::Mat& oPolyPts_THERMAL, cv::Mat& oPolyPts_VISIBLE) {
            oPolyPts_THERMAL.create(3,6,CV_64FC1);
            oPolyPts_VISIBLE.create(3,6,CV_64FC1);
            for(int i=0; i<oPolyListMat_THERMAL.rows; ++i) {
                for(int j=0; j<oPolyListMat_VISIBLE.cols; ++j) {
                    oPolyPts_VISIBLE.at<double>(0,i*oPolyListMat_VISIBLE.cols+j) = oPolyListMat_VISIBLE.at<cv::Vec2d>(i,j).val[0];
                    oPolyPts_VISIBLE.at<double>(1,i*oPolyListMat_VISIBLE.cols+j) = oPolyListMat_VISIBLE.at<cv::Vec2d>(i,j).val[1];
                    oPolyPts_VISIBLE.at<double>(2,i*oPolyListMat_VISIBLE.cols+j) = 1;
                    oPolyPts_THERMAL.at<double>(0,i*oPolyListMat_THERMAL.cols+j) = oPolyListMat_THERMAL.at<cv::Vec2d>(i,j).val[0];
                    oPolyPts_THERMAL.at<double>(1,i*oPolyListMat_THERMAL.cols+j) = oPolyListMat_THERMAL.at<cv::Vec2d>(i,j).val[1];
                    oPolyPts_THERMAL.at<double>(2,i*oPolyListMat_THERMAL.cols+j) = 1;
                }
            }
        }

        void DrawPolyPtsMatsToMat(const cv::Mat& oPolyListMat_THERMAL, const cv::Mat& oPolyListMat_VISIBLE, cv::Mat& oPolyMat_THERMAL, cv::Mat& oPolyMat_VISIBLE) {
            CV_Assert(!oPolyMat_THERMAL.empty() && !oPolyMat_VISIBLE.empty());
            for(int i=0; i<oPolyListMat_THERMAL.rows; ++i) {
                cv::Mat temp;
                oPolyListMat_THERMAL.row(i).convertTo(temp,CV_32SC2);
                cv::fillConvexPoly(oPolyMat_THERMAL,temp,cv::Scalar_<uchar>::all(255));
                oPolyListMat_VISIBLE.row(i).convertTo(temp,CV_32SC2);
                cv::fillConvexPoly(oPolyMat_VISIBLE,temp,cv::Scalar_<uchar>::all(255));
            }
        }

        int OpenTestSeqVideos(cv::VideoCapture& oCapOrig_THERMAL, cv::VideoCapture& oCapBGS_THERMAL, cv::VideoCapture& oCapOrig_VISIBLE, cv::VideoCapture& oCapBGS_VISIBLE, int sed_ID) {
			
			std::stringstream ss;
			ss << sed_ID;
			std::string sed_ID_str = ss.str()+".avi";


            oCapOrig_THERMAL.open(LITIV2012_TEST_SET_PATH "\\THERMAL\\input\\"+sed_ID_str);
            oCapBGS_THERMAL.open(LITIV2012_TEST_SET_BGS_PATH "\\THERMAL\\"+sed_ID_str);
            oCapOrig_VISIBLE.open(LITIV2012_TEST_SET_PATH "\\VISIBLE\\input\\"+sed_ID_str);
            oCapBGS_VISIBLE.open(LITIV2012_TEST_SET_BGS_PATH "\\VISIBLE\\"+sed_ID_str);
			//oCapOrig_THERMAL.open(LITIV2012_TEST_SET_PATH "\\THERMAL\\input\\in%06d.jpg");
			//oCapBGS_THERMAL.open(LITIV2012_TEST_SET_BGS_PATH "\\THERMAL\\bin%06d.png");
			//oCapOrig_VISIBLE.open(LITIV2012_TEST_SET_PATH "\\VISIBLE\\input\\in%06d.jpg");
			//oCapBGS_VISIBLE.open(LITIV2012_TEST_SET_BGS_PATH "\\VISIBLE\\bin%06d.png");

            if(!oCapOrig_THERMAL.isOpened() || !oCapOrig_VISIBLE.isOpened() || !oCapBGS_THERMAL.isOpened() || !oCapBGS_VISIBLE.isOpened())
                CV_Error(-1,"failed to open video sequences");
            if((int)oCapOrig_THERMAL.get(cv::CAP_PROP_FRAME_COUNT)!=(int)oCapOrig_VISIBLE.get(cv::CAP_PROP_FRAME_COUNT) ||
                (int)oCapBGS_THERMAL.get(cv::CAP_PROP_FRAME_COUNT)!=(int)oCapBGS_VISIBLE.get(cv::CAP_PROP_FRAME_COUNT) ||
                (int)oCapBGS_THERMAL.get(cv::CAP_PROP_FRAME_COUNT)!=(int)oCapOrig_VISIBLE.get(cv::CAP_PROP_FRAME_COUNT))
                CV_Error(-1,"thermal/visible sequence frame count mismatch");

			cv::Mat oTempImg_THERMAL, oTempImg_VISIBLE;
            oCapOrig_THERMAL >> oTempImg_THERMAL;
            oCapOrig_VISIBLE >> oTempImg_VISIBLE;
            if(oTempImg_THERMAL.empty() || oTempImg_VISIBLE.empty())
                CV_Error(-1,"failed to fetch init frames from orig video sequences");
            const cv::Size oInputSize_THERMAL = oTempImg_THERMAL.size();
            const cv::Size oInputSize_VISIBLE = oTempImg_VISIBLE.size();
            oCapBGS_THERMAL >> oTempImg_THERMAL;
            oCapBGS_VISIBLE >> oTempImg_VISIBLE;
            if(oTempImg_THERMAL.empty() || oTempImg_VISIBLE.empty())
                CV_Error(-1,"failed to fetch init frames from bgs video sequences");
            if(oInputSize_THERMAL!=oTempImg_THERMAL.size() || oInputSize_VISIBLE!=oTempImg_VISIBLE.size())
                CV_Error(-1,"orig/bgs frame size mismatch");
			oCapOrig_THERMAL.set(cv::CAP_PROP_POS_FRAMES,0);
            oCapOrig_VISIBLE.set(cv::CAP_PROP_POS_FRAMES,0);
            oCapBGS_THERMAL.set(cv::CAP_PROP_POS_FRAMES,0);
            oCapBGS_VISIBLE.set(cv::CAP_PROP_POS_FRAMES,0);
			//CV_Assert((int)oCapOrig_THERMAL.get(cv::CAP_PROP_FRAME_COUNT));
            return (int)oCapOrig_THERMAL.get(cv::CAP_PROP_FRAME_COUNT);
        }

    }; // namespace LITIV2012

};// namespace DatasetUtils

