#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib>
#include <string>
#include <cmath>
#include <unistd.h>

using namespace cv;
using namespace std;


float sGauss(float z, int sig){
    z = - (z*z);
    z = z / (2.0 * (sig*sig));
    z = exp(z);
    return z;
}

float gGauss(float z, int sig){
    z = - (z*z);
    z = z / (2.0 * (sig*sig));
    z = exp(z);
    z = z / (2 * 3.14 * (float)sig * (float)sig);
    return z;
}


Mat addNoise(Mat input, int prob){


    //Prob is a probability parameter with range between 1-100

    Mat output = input.clone();

    for(int i = 0; i < output.rows; i++){
        for(int j = 0; j < output.cols; j++){
        

            int r = (rand() % 100) + 1;
            if(r < prob){
                output.at<uchar>(i, j) = 0;
            }
            else if(r > (100-prob)){
                output.at<uchar>(i, j) = 255;
            }


        }
        
    }

    return output;
}

Mat dilation(Mat input, int window){

    Mat output = input.clone();

    int offset = 1;

    if(window > 2){
        offset = window / 2;
    }   


    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);

            int maxV = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){
                    if(input.at<uchar>(k, l) > maxV){
                        maxV = input.at<uchar>(k, l);
                    }
                }
            }

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){                    
                    output.at<uchar>(k, l) = maxV;
                }
            }

        }
    }

    return output;
}

Mat erosion(Mat input, int window){

    Mat output = input.clone();

    int offset = 1;

    if(window > 2){
        offset = window / 2;
    }   


    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);

            int minV = 255;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){
                    if(input.at<uchar>(k, l) < minV){
                        minV = input.at<uchar>(k, l);
                    }
                }
            }

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){                    
                    output.at<uchar>(k, l) = minV;
                }
            }

        }
    }

    return output;
}



Mat harrisCornerDetection(Mat input, float thres){

    Mat output = input.clone();
    Mat xDif = input.clone();
    Mat yDif = input.clone();


    int xKernel[3][3] = {
    {1, 0, -1},
    {2, 0, -2},
    {1, 0, -1}
    };

    int yKernel[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
    };

    int offset = 1;


    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);


            int gX = 0, gY = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){

                    gX += input.at<uchar>(k, l) * xKernel[k-mink][l-minl];
                    gY += input.at<uchar>(k, l) * yKernel[k-mink][l-minl];

                }
            }

            xDif.at<uchar>(i, j) = gX;
            yDif.at<uchar>(i, j) = gY;

        }
    }


    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);

            float m11 = 0, m12 = 0, m21 = 0, m22 = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){

                    m11 += (xDif.at<uchar>(k, l) * xDif.at<uchar>(k, l));
                    m12 += (xDif.at<uchar>(k, l) * yDif.at<uchar>(k, l));
                    m22 += (yDif.at<uchar>(k, l) * yDif.at<uchar>(k, l));

                }
            }

            m21 = m12;

            Mat sTemp = (Mat_<float>(2, 2) << m11, m12, m21, m22);
            Mat E, V;

            eigen(sTemp, E, V);

            float l1 = E.at<float>(0, 0);
            float l2 = E.at<float>(0, 1);

            float R = (l1 * l2) / ((l1 + l2) * 10000);
            //float R = (l1 * l2) - (0.04 * (l1 + l2) * (l1 + l2));

            output.at<uchar>(i, j) = 0;

            if(R > thres){
                output.at<uchar>(i, j) = 255;
            }

        }
    }
    

    return output;

}


Mat sobelEdge(Mat input){

    Mat output = input.clone();


    int xKernel[3][3] = {
    {1, 0, -1},
    {2, 0, -2},
    {1, 0, -1}
    };

    int yKernel[3][3] = {
    {1, 2, 1},
    {0, 0, 0},
    {-1, -2, -1}
    };

    int offset = 1;


    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);


            int gX = 0, gY = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){

                    gX += input.at<uchar>(k, l) * xKernel[k-mink][l-minl];
                    gY += input.at<uchar>(k, l) * yKernel[k-mink][l-minl];

                }
            }

            output.at<uchar>(i, j) = sqrt((gX * gX) + (gY * gY));

        }
    }



    return output;

}



Mat fastAdaptiveBinarize(Mat input, int size){

    Mat output = input.clone();

    //Offset affects the area under consideration
    int offset = size;

    int counter = offset;

    int mean = 100;

    //Computing localized threshold using mean
    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int sum = 0;
            int count = 0;            

            if(counter == offset){

                int mink = max(0, i-offset);
                int minl = max(0, j-offset);
                int maxk = min(input.rows, i+offset+1);
                int maxl = min(input.cols, j+offset+1);

                for(int k = mink; k < maxk; k++){
                    for(int l = minl; l < maxl; l++){

                        sum += input.at<uchar>(k, l); 
                        count++;

                    }
                }

                mean = sum / count;
            }

            if(input.at<uchar>(i, j) >= mean){
                output.at<uchar>(i, j) = 255;
            }
            else{
                output.at<uchar>(i, j) = 0;
            }

            counter--;
            if(counter <= 0){
                counter = offset;
            }

        }
    }    

    return output;

}





Mat adaptiveBinarize(Mat input, int size){

    Mat output = input.clone();

    //Offset affects the area under consideration
    int offset = size;

    //Computing localized threshold using mean
    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);

            int sum = 0;
            int count = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){

                    sum += input.at<uchar>(k, l); 
                    count++;

                }
            }

            int mean = sum / count;

            if(input.at<uchar>(i, j) >= mean){
                output.at<uchar>(i, j) = 255;
            }
            else{
                output.at<uchar>(i, j) = 0;
            }

        }
    }    

    return output;

}





Mat scaledBilateralGauss(Mat input, int r, int s, int g, int ws){

    Mat output = input.clone();
    Mat scaledG = input.clone();

    if(ws > 7) ws = 7;
    if(ws < 3) ws = 3;
    if(ws%2==0) ws -= 1;

    int offset = ws/2;



    //Computing Scaled Guassian Mat
    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);

            float ig = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){

                    ig += (input.at<uchar>(k, l) * gGauss((float)max(abs(i-k), abs(j-l)), g));

                }
            }

            scaledG.at<uchar>(i, j) = ig;

        }
    }
    


    //Applying on Main Image
    for(int i = 0; i < input.rows; i++){
        for(int j = 0; j < input.cols; j++){

            int mink = max(0, i-offset);
            int minl = max(0, j-offset);
            int maxk = min(input.rows, i+offset+1);
            int maxl = min(input.cols, j+offset+1);


            float sum1 = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){

                    float tempm = 1;

                    float md = max(abs(k-i), abs(l-j));

                    tempm *= sGauss(md, s);
                    tempm *= input.at<uchar>(k, l);

                    float gr = sGauss(fabs(scaledG.at<uchar>(i, j) - input.at<uchar>(i, j)), r);

                    tempm *= gr;   
                    sum1 += tempm;                

                }
            }

            float sum2 = 0;

            for(int k = mink; k < maxk; k++){
                for(int l = minl; l < maxl; l++){

                    float tempm = 1;

                    float md = max(abs(k-i), abs(l-j));

                    tempm *= sGauss(md, s);

                    int minm = max(0, k-offset);
                    int minn = max(0, l-offset);
                    int maxm = min(input.rows, k+offset+1);
                    int maxn = min(input.cols, l+offset+1);

                    float gr = sGauss(fabs(scaledG.at<uchar>(i, j) - input.at<uchar>(i, j)), r);

                    tempm *= gr;    
                    sum2 += tempm;                

                }
            }

            output.at<uchar>(i, j) = sum1 / sum2;

        }
    }

    return output;

}



int main( int argc, char** argv )
{
    String imageName("cavepainting1.JPG");
    if( argc > 1)
    {
        imageName = argv[1];
    }

    Mat image;
    image = imread(imageName, IMREAD_COLOR ); 
    if( image.empty() )        
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Mat gray_image;
    cvtColor( image, gray_image, COLOR_BGR2GRAY );

    while(1){

        system("clear");

        int input = 0;
        cout << "Please select the operation to perform: (Enter Number) " << endl;
        cout << "1. Original Image" << endl;
        cout << "2. Grayscaled Image" << endl;
        cout << "3. Denoised Image by Scaled Bilateral Guassian Filter" << endl;
        cout << "4. Sharpened Image" << endl;
        cout << "5. Edge detection" << endl;
        cout << "6. Binary image (using adaptive thresholding)" << endl;
        cout << "7. Harris Corner Detection" << endl;
        cout << "9. Morphological Transforms" << endl;
        cout << "0. Exit" << endl << endl;
        cout << "Select Operation: ";

        cin >> input;

        if(input == 1){
            namedWindow( "Cave Painting", WINDOW_AUTOSIZE );
            imshow( "Cave Painting", image);
            waitKey(0);
        }

        else if(input == 2){
            namedWindow( "Gray image", WINDOW_AUTOSIZE );
            imshow( "Gray image", gray_image);  
            waitKey(0);
        }

        else if(input == 3){

            int s = 4, r = 12, g = 3, ws = 3;

            cout << "Enter parameters SigS, SigR and SigG (in that order, e.g. \"3 12 4\"): ";
            cin >> s >> r >> g;

            //Last argument is the overall amount of noise of add. pass argument between 1 - 100
            Mat noise_image = addNoise(gray_image, 5);

            //Last Argument is window size. Can be tweaked for better results
            Mat image_sbg = scaledBilateralGauss(noise_image, r, s, g, 5);

            Mat checkGauss;

            hconcat(noise_image, image_sbg, checkGauss);

            cout << "Opening Scaled Bilateral Gaussian Filtered image in new window." << endl;

            namedWindow( "Left: Image + Noise   Right: FIltered Image", WINDOW_AUTOSIZE );
            imshow( "Left: Image + Noise   Right: FIltered Image", checkGauss); 
            waitKey(0);

        }

        else if(input == 4){

            //Image sharpening by amplifying finer details in the image (Assuming low amount of noise)
            //The first parameter controls the sharpening for finer details.
            Mat image_coarse = scaledBilateralGauss(gray_image, 12, 5, 2, 5);
            Mat fine_Details = gray_image - image_coarse;
            Mat damp_Details = 255 - fine_Details - image_coarse;

            //Multiplying by 0.1 to amplify finer details by 10%. In practical applications this can be set by user
            Mat sharp_Image = gray_image  + (fine_Details) - (0.2 * damp_Details);

            Mat sharpComp;
            hconcat(gray_image, sharp_Image, sharpComp);

            cout << "Opening sharpened image (using bilateral Gaussian to enhance fine details) in new window." << endl;

            namedWindow( "Sharpened image", WINDOW_AUTOSIZE );
            imshow( "Sharpened image", sharpComp); 
            waitKey(0);

        }

        else if(input == 5){

            //Edge detection using sobel operator by combining gradients on X and Y axis
            Mat blurImg;            
            blur( gray_image, blurImg, Size(5, 5) );

            //using only Sobel operator is not a great idea (sobel + hysteresis thresholding)
            //Harris Response is also a decent metric for edge detection (with lower threshold values to show edges). Again canny is better
            //Mat edgeMat = harrisCornerDetection(blurImg, 5);

            /* Uncomment to use canny
            int t1 = 10, t2 = 40;            
            cout << "Enter Canny Thresholds (Threshold 1 and 2 in order. e.g. \"10 60\"): ";
            cin >> t1 >> t2;            
            Canny(blurImg, blurImg, t1, t2, 3);
            */


            Mat sobelMat = sobelEdge(blurImg);
            Mat binaryMat = fastAdaptiveBinarize(sobelMat, 10);
            Mat edgeMat = erosion(binaryMat, 2);

            Mat edgeComp;
            hconcat(gray_image, edgeMat, edgeComp);

            namedWindow( "Edge Detection image", WINDOW_AUTOSIZE );

            imshow("Edge Detection image", edgeComp);
            waitKey(0);            
            
        }

        else if(input == 6){

            //Adaptive thresholding using localized mean method (over a area of given size)

            Mat binaryMat = adaptiveBinarize(gray_image, 10);
            
            Mat binaryComp;
            hconcat(gray_image, binaryMat, binaryComp);

            cout << "Opening binary image (using adaptive thresholding) in new window." << endl;

            namedWindow( "Binary image", WINDOW_AUTOSIZE );
            imshow( "Binary image", binaryComp); 
            waitKey(0);           
            
        }

        else if(input == 7){

            //Harris Corner Point Detection

            float thres;            
            cout << "Enter Harris Threshold (Between 1 - 20, Lower Values show edges. Higher values show corners): ";
            cin >> thres;

            Mat blurImg;
            blur( gray_image, blurImg, Size(3, 3) );

            Mat cornerMat = harrisCornerDetection(blurImg, thres);
            
            Mat cornerComp;
            hconcat(gray_image, cornerMat, cornerComp);

            cout << "Opening corners detected Image in new window." << endl;

            namedWindow( "Binary image", WINDOW_AUTOSIZE );
            imshow( "Binary image", cornerComp); 
            waitKey(0);           
            
        }


        else if(input == 9){

            //Morphological Operations

            int choice;            
            cout << "Enter 1 for Dilation, 2 for Erosion, 3 for Opening, 4 for Closing: ";
            cin >> choice;

            if(choice == 1){

                cout << "Enter Strength (0-10): ";
                int strength;
                cin >> strength;

                Mat result = dilation(gray_image, strength);

                Mat cornerComp;
                hconcat(gray_image, result, cornerComp);

                cout << "Opening corners detected Image in new window." << endl;

                namedWindow( "Binary image", WINDOW_AUTOSIZE );
                imshow( "Binary image", cornerComp); 
                waitKey(0); 

            }   

            else if(choice == 2){

                cout << "Enter Strength (0-10): ";
                int strength;
                cin >> strength;

                Mat result = erosion(gray_image, strength);

                Mat cornerComp;
                hconcat(gray_image, result, cornerComp);

                cout << "Opening corners detected Image in new window." << endl;

                namedWindow( "Binary image", WINDOW_AUTOSIZE );
                imshow( "Binary image", cornerComp); 
                waitKey(0); 

            }   

            else if(choice == 3){

                cout << "Enter Strength (0-10): ";
                int strength;
                cin >> strength;

                Mat eroded = erosion(gray_image, strength);
                Mat result = dilation(eroded, strength);

                Mat cornerComp;
                hconcat(gray_image, result, cornerComp);

                cout << "Opening corners detected Image in new window." << endl;

                namedWindow( "Binary image", WINDOW_AUTOSIZE );
                imshow( "Binary image", cornerComp); 
                waitKey(0); 

            }   

            else if(choice == 4){

                cout << "Enter Strength (0-10): ";
                int strength;
                cin >> strength;

                Mat dilated = dilation(gray_image, strength);
                Mat result = erosion(dilated, strength);

                Mat cornerComp;
                hconcat(gray_image, result, cornerComp);

                cout << "Opening corners detected Image in new window." << endl;

                namedWindow( "Binary image", WINDOW_AUTOSIZE );
                imshow( "Binary image", cornerComp); 
                waitKey(0); 

            }   

            else{
                cout << "Input Not Recognized" << endl;
            }              
            
        }


        else if(input == 0) {
            break;
        }

        else{
            continue;
        }


    }

    return 0;
}