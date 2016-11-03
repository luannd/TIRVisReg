//! dataset/application parameters
#define LITIV2012_DATASET_PATH          "D:\\0_Polytechnique_Montreal2015\\Dataset\\Luan_eval\\"
//#define LITIV2012_DATASET_PATH          "C:\\CodePartage\\Luan\\litiv2012_dataset\\"

#define LITIV2012_TEST_SEQUENCE_ID      4 // [1..9]
#include "LITIV2012Utils.h"             // automatically adds all other required LITIV dataset defines
#define RESULTS_PATH                    LITIV2012_DATASET_PATH "results\\"
#define USE_FULL_DEBUG_DISPLAY          1
#define USE_THERMAL_TO_VISIBLE_PROJ     1
#define USE_FILESTORAGE_RES_OUTPUT      0
#define USE_VIDEOWRITER_RES_OUTPUT      0

#if (USE_VIDEOWRITER_RES_OUTPUT&&!USE_FULL_DEBUG_DISPLAY)
#error "cannot write video output without debug display on"
#endif //(USE_VIDEOWRITER_RES_OUTPUT&&!USE_FULL_DEBUG_DISPLAY)
#if (USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)
#define RES_OUTPUT_FILE_PREFIX "test_"
#endif //(USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)
#if (USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)
#define RES_OUTPUT_FILE_PREFIX_FULL RESULTS_PATH RES_OUTPUT_FILE_PREFIX "seq" __LITIV_STR(LITIV2012_TEST_SEQUENCE_ID)
#if USE_VIDEOWRITER_RES_OUTPUT
#define VIDEOWRITER_OUTPUT_FILE_PATH RES_OUTPUT_FILE_PREFIX_FULL ".avi"
#endif //USE_VIDEOWRITER_RES_OUTPUT
#if USE_FILESTORAGE_RES_OUTPUT
#define FILESTORAGE_OUTPUT_FILE_PATH RES_OUTPUT_FILE_PREFIX_FULL ".yml"
#endif //USE_FILESTORAGE_RES_OUTPUT
#endif //(USE_VIDEOWRITER_RES_OUTPUT||USE_FILESTORAGE_RES_OUTPUT)

#include "MultimodalVideoRegistrAlg.h"

using namespace cv;

#include <time.h>
using namespace std;

void Myeval(MultimodalVideoRegistrAlg oAlg);

clock_t Tbegin;

int main() {

    try {
		 
		MultimodalVideoRegistrAlg oAlg, oAlg2, oAlg3;

        cv::Mat oGTTransMat_THERMAL,oGTTransMat_VISIBLE;
        cv::Mat oPolyListMat_THERMAL,oPolyListMat_VISIBLE;

		

		//Convert from Matrix to List
        cv::Mat oPolyPts_THERMAL,oPolyPts_VISIBLE;
        DatasetUtils::LITIV2012::ConvertPolyPtsMatsToPtsLists(oPolyListMat_THERMAL,oPolyListMat_VISIBLE,oPolyPts_THERMAL,oPolyPts_VISIBLE);

		//Calculate Number of frames in video
        cv::VideoCapture oCapOrig_THERMAL,oCapBGS_THERMAL,oCapOrig_VISIBLE,oCapBGS_VISIBLE;
        const int nFrameCount = DatasetUtils::LITIV2012::OpenTestSeqVideos(oCapOrig_THERMAL,oCapBGS_THERMAL,oCapOrig_VISIBLE,oCapBGS_VISIBLE,LITIV2012_TEST_SEQUENCE_ID);
        std::cout << "(" << nFrameCount << " frames total)" << std::endl;

		

		//Input frames
        cv::Mat oTempImg_THERMAL, oTempImg_VISIBLE;
        oCapOrig_THERMAL >> oTempImg_THERMAL; oCapOrig_THERMAL.set(cv::CAP_PROP_FRAME_COUNT,0);
        oCapOrig_VISIBLE >> oTempImg_VISIBLE; oCapOrig_VISIBLE.set(cv::CAP_PROP_FRAME_COUNT,0);

        const cv::Size oInputSize_THERMAL = oTempImg_THERMAL.size();
        const cv::Size oInputSize_VISIBLE = oTempImg_VISIBLE.size();
        cv::Mat oPolyMat_THERMAL(oInputSize_THERMAL,CV_8UC1,cv::Scalar_<uchar>::all(0)),oPolyMat_VISIBLE(oInputSize_VISIBLE,CV_8UC1,cv::Scalar_<uchar>::all(0));
        DatasetUtils::LITIV2012::DrawPolyPtsMatsToMat(oPolyListMat_THERMAL,oPolyListMat_VISIBLE,oPolyMat_THERMAL,oPolyMat_VISIBLE);

        cv::Mat oSource_THERMAL,oSource_VISIBLE;
        cv::Mat oForeground_THERMAL,oForeground_VISIBLE;
        cv::Mat oContours_THERMAL,oContours_VISIBLE;
#if USE_THERMAL_TO_VISIBLE_PROJ
#if USE_FULL_DEBUG_DISPLAY
        const cv::Mat& oSource_ToTransform = oSource_THERMAL;
        //const cv::Mat& oSource = oSource_VISIBLE;
        cv::Mat& oContours_ToTransform = oContours_THERMAL;
        cv::Mat& oContours = oContours_VISIBLE;
#endif //USE_FULL_DEBUG_DISPLAY
        const cv::Size& oTransformedImageSize = oInputSize_VISIBLE;
        const cv::Mat& oPolyMat_ToTransform = oPolyMat_THERMAL;
        const cv::Mat& oPolyMat = oPolyMat_VISIBLE;
        const cv::Mat& oPolyPts_ToTransform = oPolyPts_THERMAL;
        const cv::Mat& oPolyPts = oPolyPts_VISIBLE;
        const cv::Mat& oGTTransMat = oGTTransMat_THERMAL;
        const cv::Mat& oGTTransMat_inv = oGTTransMat_VISIBLE;
#else //!USE_THERMAL_TO_VISIBLE_PROJ
#if USE_FULL_DEBUG_DISPLAY
        const cv::Mat& oSource_ToTransform = oSource_VISIBLE;
        //const cv::Mat& oSource = oSource_THERMAL;
        cv::Mat& oContours_ToTransform = oContours_VISIBLE;
        cv::Mat& oContours = oContours_THERMAL;
#endif //USE_FULL_DEBUG_DISPLAY
        const cv::Size& oTransformedImageSize = oInputSize_THERMAL;
        const cv::Mat& oPolyMat_ToTransform = oPolyMat_VISIBLE;
        const cv::Mat& oPolyMat = oPolyMat_THERMAL;
        const cv::Mat& oPolyPts_ToTransform = oPolyPts_VISIBLE;
        const cv::Mat& oPolyPts = oPolyPts_THERMAL;
        const cv::Mat& oGTTransMat = oGTTransMat_VISIBLE;
        const cv::Mat& oGTTransMat_inv = oGTTransMat_THERMAL;
#endif //!USE_THERMAL_TO_VISIBLE_PROJ

        cv::Mat oTransformedSource(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oGTTransformedSource(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oTransformedContours(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oGTTransformedContours(oTransformedImageSize,CV_8UC3,cv::Scalar_<uchar>::all(0)); // debug
        cv::Mat oTransformedPolyMat(oTransformedImageSize,CV_8UC1,cv::Scalar_<uchar>::all(0)); // eval
        cv::Mat oGTTransformedPolyMat(oTransformedImageSize,CV_8UC1,cv::Scalar_<uchar>::all(0)); // eval

		//cv::warpPerspective(oPolyMat_ToTransform, oGTTransformedPolyMat, oGTTransMat, oTransformedImageSize, cv::INTER_NEAREST | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
        
		//Calculate overlap error
		//const float fGTPolyOverlapError = DatasetUtils::CalcForegroundOverlapError(oPolyMat,oGTTransformedPolyMat);
  //      const cv::Mat oGTTransformedPolyPts = oGTTransMat_inv*oPolyPts_ToTransform;
  //      const cv::Point2d oGTPolyRegError = DatasetUtils::CalcPolyRegError(oPolyPts,oGTTransformedPolyPts);

#if USE_VIDEOWRITER_RES_OUTPUT
        cv::VideoWriter oResultWriter(VIDEOWRITER_OUTPUT_FILE_PATH,cv::VideoWriter::fourcc('M','J','P','G'),15,cv::Size(oTransformedImageSize.width*3,oTransformedImageSize.height*2));
#endif //USE_VIDEOWRITER_RES_OUTPUT
#if USE_FILESTORAGE_RES_OUTPUT
        cv::FileStorage oResultFS(FILESTORAGE_OUTPUT_FILE_PATH,cv::FileStorage::WRITE);
        oResultFS << "config" << "{";
        oResultFS << "USE_THERMAL_TO_VISIBLE_PROJ" << USE_THERMAL_TO_VISIBLE_PROJ;
        oResultFS << "}";
        oResultFS << "results" << "[";
#endif //USE_FILESTORAGE_RES_OUTPUT

#if USE_FULL_DEBUG_DISPLAY
        bool bContinuousUpdates = false;
#endif //USE_FULL_DEBUG_DISPLAY
        int nFirstIndex = nFrameCount;
        cv::Point2d oCumulativePolyRegErrors(0,0);
        float fCumulativePolyOverlapErrors = 0.0f;

		cv::VideoCapture oCapOrig_THERMAL_tmp = oCapOrig_THERMAL, 
						 oCapBGS_THERMAL_tmp = oCapBGS_THERMAL, 
						 oCapOrig_VISIBLE_tmp = oCapOrig_VISIBLE, 
						 oCapBGS_VISIBLE_tmp = oCapBGS_VISIBLE;
		DatasetUtils::LITIV2012::OpenTestSeqVideos(oCapOrig_THERMAL_tmp, oCapBGS_THERMAL_tmp, oCapOrig_VISIBLE_tmp, oCapBGS_VISIBLE_tmp, LITIV2012_TEST_SEQUENCE_ID);
		//Them-Luan //Tinh truoc de co matran tot nhat
		Size tmSize;
		std::cout << "Initialize ";
		for (int nCurrFrameIndex = 0; nCurrFrameIndex < nFrameCount; ++nCurrFrameIndex){
			std::cout << "# Tinh trc: " << nCurrFrameIndex << std::endl;
			std::cout << ".";
			cv::Mat oForegroundBGR_THERMAL, oForegroundBGR_VISIBLE;
			oCapBGS_THERMAL_tmp >> oForegroundBGR_THERMAL;
			oCapBGS_VISIBLE_tmp >> oForegroundBGR_VISIBLE;
			if (oForegroundBGR_THERMAL.empty() || oForegroundBGR_VISIBLE.empty())
				break;

			cv::cvtColor(oForegroundBGR_THERMAL, oForeground_THERMAL, COLOR_BGR2GRAY);
			cv::cvtColor(oForegroundBGR_VISIBLE, oForeground_VISIBLE, COLOR_BGR2GRAY);

			Size size(oForeground_THERMAL.cols/2, oForeground_THERMAL.rows/2);//de size mac dinh -> nhanh tg
			tmSize = size;
			resize(oForeground_THERMAL, oForeground_THERMAL, size);//resize image
			resize(oForeground_VISIBLE, oForeground_VISIBLE, size);//resize image

			oAlg2.ProcessForeground(oForeground_THERMAL, oForeground_VISIBLE);



			//them tmp
			//break;
		}
		std::cout << endl;
		//Mat luantmp = oAlg2.GetTransformationMatrix(false);
		//Thu set matran nhan gia tri ban dau, thay vi ma tran don vi
		//oAlg.setTransMat(oAlg2);
		oAlg2.iCountFrame = 0;
		oAlg2.minisBad = oAlg2.minisBadFundamental = 100;

		oAlg2.isRunSecond = true;

		while (!oAlg2.m_queueFrames.empty())
			oAlg2.m_queueFrames.pop();
		oAlg = oAlg2;
		
		cv::VideoCapture oCapOrig_THERMAL_tmp2 = oCapOrig_THERMAL,
			oCapBGS_THERMAL_tmp2 = oCapBGS_THERMAL,
			oCapOrig_VISIBLE_tmp2 = oCapOrig_VISIBLE,
			oCapBGS_VISIBLE_tmp2 = oCapBGS_VISIBLE;
		DatasetUtils::LITIV2012::OpenTestSeqVideos(oCapOrig_THERMAL_tmp2, oCapBGS_THERMAL_tmp2, oCapOrig_VISIBLE_tmp2, oCapBGS_VISIBLE_tmp2, LITIV2012_TEST_SEQUENCE_ID);

		

		//freopen("vid2GT.txt", "r", stdin);
		//freopen("vid2Res.txt", "w", stdout);

		for (int nCurrFrameIndex = 0; nCurrFrameIndex < nFrameCount; ++nCurrFrameIndex){
			//int id = -1, dGT, y, x, num;
			//cin >> id >> dGT >> y >> x >> num;
			//if (id < 0)
			//	break;
			//std::cout << "# Eval:" << nCurrFrameIndex << std::endl;

			cv::Mat oForegroundBGR_THERMAL, oForegroundBGR_VISIBLE;
			oCapBGS_THERMAL_tmp2 >> oForegroundBGR_THERMAL;
			oCapBGS_VISIBLE_tmp2 >> oForegroundBGR_VISIBLE;
			if (oForegroundBGR_THERMAL.empty() || oForegroundBGR_VISIBLE.empty())
				break;

			cv::cvtColor(oForegroundBGR_THERMAL, oForeground_THERMAL, COLOR_BGR2GRAY);
			cv::cvtColor(oForegroundBGR_VISIBLE, oForeground_VISIBLE, COLOR_BGR2GRAY);

			
			resize(oForeground_THERMAL, oForeground_THERMAL, tmSize);//resize image
			resize(oForeground_VISIBLE, oForeground_VISIBLE, tmSize);//resize image

			oAlg2.disparityFinal.clear();

			oAlg2.ProcessForeground(oForeground_THERMAL, oForeground_VISIBLE);

			//int numEval = 0;

			//for (int i = x; i < x + num; i++){
			//	if (y >= 0 && y < oAlg2.beforeH2.rows && i + dGT >= 0 && i + dGT < oAlg2.beforeH2.cols)
			//	if (oAlg2.beforeH2.at<uchar>(y, i + dGT) > 0)
			//		numEval++;
			//}

			//imshow("final", oAlg2.afterH1 / 2 + oForeground_THERMAL/2);
			cv::waitKey(0);
			//int trueNum = 0;
			//for (int i = x; i < x + num; i++){
			//	if (y >= 0 && y < oAlg2.afterH1.rows && i >= 0 && i< oAlg2.afterH1.cols)
			//		if (oAlg2.afterH1.at<uchar>(y, i) > 0)
			//			trueNum++;
			//}

			//num = trueNum;

			//int finalDis = INT_MAX, errD = INT_MAX;
			//for (int i = 0; i < oAlg2.disparityFinal.size(); i++){
			//	if (errD > abs(abs(oAlg2.disparityFinal[i]) - dGT)){
			//		finalDis = abs(oAlg2.disparityFinal[i]);
			//		errD = abs(abs(oAlg2.disparityFinal[i]) - dGT);
			//	}
			//}


			//cout << id << " " << dGT <<" "<< finalDis << endl << trueNum << " " << numEval << endl;
		}


//
//
//        for(int nCurrFrameIndex=0; nCurrFrameIndex<nFrameCount; ++nCurrFrameIndex) {
//            //if((nCurrFrameIndex%50)==0)
//                std::cout << "# " << nCurrFrameIndex << std::endl;
//				
//            oCapOrig_THERMAL >> oSource_THERMAL;
//            oCapOrig_VISIBLE >> oSource_VISIBLE;
//            if(oSource_THERMAL.empty() || oSource_VISIBLE.empty())
//                break;
//
//            cv::Mat oForegroundBGR_THERMAL,oForegroundBGR_VISIBLE;
//            oCapBGS_THERMAL >> oForegroundBGR_THERMAL;
//            oCapBGS_VISIBLE >> oForegroundBGR_VISIBLE;
//            if(oForegroundBGR_THERMAL.empty() || oForegroundBGR_VISIBLE.empty())
//                break;
//
//			
//			////Deal with images
//			//oForeground_THERMAL = oForegroundBGR_THERMAL;
//			//oForeground_VISIBLE = oForegroundBGR_VISIBLE;
//			cv::cvtColor(oForegroundBGR_THERMAL, oForeground_THERMAL, COLOR_BGR2GRAY);
//			cv::cvtColor(oForegroundBGR_VISIBLE, oForeground_VISIBLE, COLOR_BGR2GRAY);
//
//			Size size(oForeground_THERMAL.rows/3, oForeground_THERMAL.cols/3);//the dst image size,e.g.100x100
//			resize(oForeground_THERMAL, oForeground_THERMAL, size);//resize image
//			resize(oForeground_VISIBLE, oForeground_VISIBLE, size);//resize image
//
//	        oAlg.ProcessForeground(oForeground_THERMAL, oForeground_VISIBLE);
//			
//			
//			const cv::Mat& oTransMat = oAlg.GetTransformationMatrix(false);
//			
//            if(!oTransMat.empty()) {
//                if(nFirstIndex==nFrameCount)
//                    nFirstIndex = nCurrFrameIndex;
//				
//                
#if USE_FULL_DEBUG_DISPLAY
//               				
//				//calculate perspective overlay
//				//if (oAlg.isUsingFundametalMatrix == false){
//				//	cv::warpPerspective(oForeground_THERMAL, oTransformedSource, oTransMat, oTransformedImageSize, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
//				//}
//				//else{
//				//	oTransformedSource = oAlg.registrationUsingFundamentalMatrix_Choose(oForeground_THERMAL);
//				//}
//
//				//if (oAlg.isNewWay == true)
//				//	oTransformedSource = oAlg.m_AdaptiveFundamentalFrame;
//
//				Mat oTransformedSource1, oTransformedSource2, oTransformedSource3;
//				float error1 = 1, error2 = 1, error3 = 1;
//				if (oAlg.isUsingFundametalMatrix == false){
//					cv::warpPerspective(oForeground_THERMAL, oTransformedSource1, oTransMat, size, cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_CONSTANT);
//					error1 = DatasetUtils::CalcForegroundOverlapError(oTransformedSource1, oForeground_VISIBLE);
//				}
//				if (oAlg.isUsingFundametalMatrix == true){
//					oTransformedSource2 = oAlg.registrationUsingFundamentalMatrix_Choose(oForeground_THERMAL);
//					error2 = DatasetUtils::CalcForegroundOverlapError(oTransformedSource2, oForeground_VISIBLE);
//				}
//				if (oAlg.isRunSecond == true){
//					oTransformedSource3 = oAlg.m_AdaptiveFundamentalFrame;
//					error3 = DatasetUtils::CalcForegroundOverlapError(oTransformedSource3, oForeground_VISIBLE);
//				}
//				if (error1 <= min(error2, error3))
//					oTransformedSource = oTransformedSource1;
//				if (error2 <= min(error1, error3))
//					oTransformedSource = oTransformedSource2;
//				if (error3 <= min(error2, error1))
//					oTransformedSource = oTransformedSource3;
//
//				float fPolyOverlapError = DatasetUtils::CalcForegroundOverlapError(oTransformedSource, oForeground_VISIBLE);
//				cout << " " << fPolyOverlapError << endl;
//				fCumulativePolyOverlapErrors += fPolyOverlapError;


				//imshow("best", oTransformedSource / 2 + oForeground_VISIBLE / 2);
				//Moi chinh
#endif //USE_FULL_DEBUG_DISPLAY				

#if USE_FILESTORAGE_RES_OUTPUT
                oResultFS << "{";
                oResultFS << "nCurrFrameIndex" << nCurrFrameIndex;
                oResultFS << "oCurrPolyRegError" << oPolyRegError;
                oResultFS << "fCurrPolyOverlapError" << fPolyOverlapError;
                oResultFS << "}";
#endif //USE_FILESTORAGE_RES_OUTPUT
//            }
#if USE_FULL_DEBUG_DISPLAY
			
            //cv::warpPerspective(oSource_ToTransform,oGTTransformedSource,oGTTransMat,oTransformedImageSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
            //cv::warpPerspective(oContours_ToTransform,oGTTransformedContours,oGTTransMat,oTransformedImageSize,cv::INTER_LINEAR|cv::WARP_INVERSE_MAP,cv::BORDER_CONSTANT);
			
			//cv::Mat oTransformedSourceOverlay = (USE_THERMAL_TO_VISIBLE_PROJ ? oSource_VISIBLE : oSource_THERMAL) + oTransformedSource;
   //         cv::putText(oTransformedSourceOverlay,"Estimated Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
			
            //cv::Mat oGTTransformedSourceOverlay = (USE_THERMAL_TO_VISIBLE_PROJ ? oSource_VISIBLE : oSource_THERMAL) + oGTTransformedSource;
            //cv::putText(oGTTransformedSourceOverlay,"GT Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
			
			//cv::Mat oTransformedContoursOverlay = (USE_THERMAL_TO_VISIBLE_PROJ ? oContours_VISIBLE : oContours_THERMAL) + oTransformedContours;
   //         cv::putText(oTransformedContoursOverlay,"Estimated Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
			//
   //         cv::Mat oGTTransformedContoursOverlay = (USE_THERMAL_TO_VISIBLE_PROJ ? oContours_VISIBLE : oContours_THERMAL) + oGTTransformedContours;
   //         cv::putText(oGTTransformedContoursOverlay,"GT Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
			//
   //         cv::Mat oTransformedPolyMatOverlay; cv::cvtColor(oPolyMat/2+oTransformedPolyMat/2,oTransformedPolyMatOverlay,cv::COLOR_GRAY2BGR);
   //         cv::putText(oTransformedPolyMatOverlay,"Estimated Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
   //         cv::Mat oGTTransformedPolyMatOverlay; cv::cvtColor(oPolyMat/2+oGTTransformedPolyMat/2,oGTTransformedPolyMatOverlay,cv::COLOR_GRAY2BGR);
   //         cv::putText(oGTTransformedPolyMatOverlay,"GT Transformation",cv::Point(20,20),cv::FONT_HERSHEY_PLAIN,0.8,cv::Scalar_<uchar>::all(255),1);
			
            //cv::Mat oTransformedOverlayRow;
            //cv::hconcat(oTransformedPolyMatOverlay,oTransformedContoursOverlay,oTransformedOverlayRow);
            //cv::hconcat(oTransformedSourceOverlay,oTransformedOverlayRow,oTransformedOverlayRow);
            //cv::Mat oGTTransformedOverlayRow;
            //cv::hconcat(oGTTransformedPolyMatOverlay,oGTTransformedContoursOverlay,oGTTransformedOverlayRow);
            //cv::hconcat(oGTTransformedSourceOverlay,oGTTransformedOverlayRow,oGTTransformedOverlayRow);
            cv::Mat oFullOverlay;
            //cv::vconcat(oTransformedOverlayRow,oGTTransformedOverlayRow,oFullOverlay);

  /*          cv::imshow("oFullOverlay",oFullOverlay);*/
   //         int nKeyPressed;
   //         if(bContinuousUpdates)
   //             nKeyPressed = cv::waitKey(1);
   //         else
   //             nKeyPressed = cv::waitKey(0);
   //         if(nKeyPressed!=-1) {
   //             nKeyPressed %= (UCHAR_MAX+1); // fixes return val bug in some opencv versions
   //             std::cout << "nKeyPressed = " << nKeyPressed%(UCHAR_MAX+1) << std::endl;
   //         }
			//if (nKeyPressed == ' '){
			//	bContinuousUpdates = !bContinuousUpdates;
			//	Tbegin = clock();
			//}
   //         else if(nKeyPressed==(int)'q')
   //             break;
#if USE_VIDEOWRITER_RES_OUTPUT
            oResultWriter.write(oFullOverlay);
#endif //USE_VIDEOWRITER_RES_OUTPUT
#endif //USE_FULL_DEBUG_DISPLAY
        //}

		//cout<<"fCumulativePolyOverlapErrors = "<<fCumulativePolyOverlapErrors/nFrameCount;
		clock_t Tend = clock();
		double elapsed_secs = double(Tend - Tbegin) / CLOCKS_PER_SEC;
		//cout << "  Time process =" << elapsed_secs;
#if USE_FILESTORAGE_RES_OUTPUT
        oResultFS << "]";
        oResultFS << "nFirstIndex" << nFirstIndex;
        oResultFS << "nFrameCount" << nFrameCount;
        oResultFS << "oAveragePolyRegError" << oAveragePolyRegError;
        oResultFS << "oGTPolyRegError" << oGTPolyRegError;
        oResultFS << "fAveragePolyOverlapError" << fAveragePolyOverlapError;
        oResultFS << "fGTPolyOverlapError" << fGTPolyOverlapError;
#endif //USE_FILESTORAGE_RES_OUTPUT
    }
    catch(const cv::Exception& err) {
        printf("cv::Exception: %s\n",err.what());
        return -1;
    }
    catch(const std::exception& err) {
        printf("std::exception: %s\n",err.what());
        return -1;
    }
    catch(...) {
        printf("unhandled exception.\n");
        return -1;
    }
    return 0;
}

void Myeval(MultimodalVideoRegistrAlg oAlg){


}