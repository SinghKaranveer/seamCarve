//
//  main.cpp
//  openCVTest
//
//  Created by Karanveer Singh on 5/23/19.
//  Copyright © 2019 Karanveer Singh. All rights reserved.
//

#include <iostream>
//#include <opencv2/imgproc/imgproc.hpp>
//#include "/usr/local/Cellar/opencv/4.1.0_2/include/opencv4/opencv2/imgproc/imgproc.hpp"
//#include "/usr/local/Cellar/opencv/4.1.0_2/include/opencv4/opencv2/core.hpp"
#include "/usr/local/Cellar/opencv/4.1.0_2/include/opencv4/opencv2/opencv.hpp"
#include <time.h>
//#include <emscripten.h>


#define HORIZONTAL 0
#define VERTICAL 1

using namespace cv;

Mat calculateImageGradient(Mat input);
Mat calculateImageEnergy(Mat input, int direction);
Mat findSeam(Mat energyMap, Mat originalImage, int direction, int iteration);
void removeSeam(Mat input, int i, int j, int location, int overHalf);

int main(int argc, const char * argv[]) {
    clock_t start = clock();
    int dir = HORIZONTAL;
    Mat testImage, outputImage, outputImage2;
    testImage = cv::imread("test3.jpg", IMREAD_COLOR);
    namedWindow("Display Window", WINDOW_AUTOSIZE);
    //imshow("Display Window", outputImage2);
    for(int i = 0; i < 200; i++)
    {
        if(i % 10 == 0)
        {
            outputImage = calculateImageGradient(testImage);
            outputImage2 = calculateImageEnergy(outputImage, dir);
        }
        testImage = findSeam(outputImage2, testImage, dir, i+1);
    }
    clock_t end = clock();
    float seconds = ((float)end - (float)start) / CLOCKS_PER_SEC;
    std::cout<<"Time elapsed "<<seconds<<std::endl;
    imwrite("test_output.jpg", testImage);
    waitKey(0);
    return 0;
}

Mat calculateImageGradient(Mat input)
{
    Mat inputBlur;
    Mat convert;
    Mat greyScale, greyX, greyY, absGreyX, absGreyY, output;
    //GaussianBlur(input, inputBlur, Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(input, greyScale, COLOR_BGR2GRAY);
    Scharr(greyScale, greyX, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT); 
    Scharr(greyScale, greyY, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT); 
    convertScaleAbs(greyX, absGreyX);
    convertScaleAbs(greyY, absGreyY);
    cv::addWeighted(absGreyX, 0.5, absGreyY, 0.5, 0, output);
    output.convertTo(convert, CV_64F, 1.0/255.0);
    return convert;
    //return absGreyX * 0.5 + absGreyY * 0.5;
    //return blue*0.33 + green*0.33 + red*0.33;
}

Mat calculateImageEnergy(Mat input, int direction)
{
    Mat output;
    int numRows, numCols, i, j;
    numRows = input.rows;
    numCols = input.cols;
    output = Mat(numRows, numCols, CV_64F, double(0));
    if(direction == VERTICAL)
    {
        double left, center, right;
        input.row(0).copyTo(output.row(0));
        for(i = 1; i < numRows; i++)
        {
            for(j = 0; j < numCols; j++)
            {
                if(j == 0)
                {
                    left = output.at<double>(i-1, 0);
                    right = output.at<double>(i-1,j+1);
                }
                else if(j == numCols - 1)
                {
                    left = output.at<double>(i-1, j-1);
                    right = output.at<double>(i-1, numCols-1);
                }
                else
                {
                    left = output.at<double>(i-1, j-1);
                    right = output.at<double>(i-1, j+1);
                }
                center = output.at<double>(i-1,j);
            
                output.at<double>(i, j) = input.at<double>(i,j) + std::min(right, std::min(center, left));
            }
        }
    }
    else  //Horizontal
    {
        double up, center, down;
        input.col(0).copyTo(output.col(0));
        for(j = 1; j < numCols; j++)
        {
            for(i = 0; i < numRows; i++)
            {
                if(i == 0)
                {
                    up = output.at<double>(0, j - 1);
                    down = output.at<double>(i + 1, j - 1);
                }
                else if(i == numRows - 1)
                {
                    up = output.at<double>(i - 1, j - 1);
                    down = output.at<double>(numRows - 1, j - 1);
                }
                else
                {
                    up = output.at<double>(i - 1, j - 1);
                    down = output.at<double>(i + 1, j - 1);
                }
                center = output.at<double>(i, j - 1);

                output.at<double>(i, j) = input.at<double>(i,j) + std::min(up, std::min(center, down));

            }
        }

    }
    Mat colorMap;
    double Cmin;
    double Cmax;
    cv::minMaxLoc(output, &Cmin, &Cmax);
    float scale = 255.0 / (Cmax - Cmin);
    output.convertTo(colorMap, CV_8U, scale);
    applyColorMap(colorMap, colorMap, cv::COLORMAP_JET);
    namedWindow("Cumulative Energy Map", WINDOW_AUTOSIZE); imshow("Cumulative Energy Map", colorMap);

    return output;
}

Mat findSeam(Mat energyMap, Mat originalImage, int direction, int iteration)
{
    int numRows, numCols, currentPoint, min;
    //std::cout<<"Iteration: "<<iteration<<std::endl;
    numRows = originalImage.rows;
    numCols = originalImage.cols;
    if(direction == VERTICAL)
    {
        cv::Point min_loc, max_loc;
        double min, max, left, center, right;
        Mat bottomRow = energyMap.row(numRows - 1);
        int overHalf = 0;
        cv::minMaxLoc(bottomRow, &min, &max, &min_loc, &max_loc);
        currentPoint = min_loc.x - iteration;
        if(currentPoint < 0)
            currentPoint = 0;
        if(currentPoint > numCols / 2)
            overHalf = 1;
        //std::cout << currentPoint << std::endl;
        removeSeam(originalImage, numRows - 1, currentPoint, direction, overHalf);
        energyMap.at<double>(numRows - 1, currentPoint) = 999;
        for(int i = numRows - 2; i > -1; i--)
        {
            //removeSeam(originalImage, i, currentPoint, direction, overHalf);
            originalImage.at<cv::Vec3b>(i, currentPoint)[0]=0; 
            originalImage.at<cv::Vec3b>(i, currentPoint)[1]=0;
            originalImage.at<cv::Vec3b>(i, currentPoint)[2]=0;
            //std::cout << i << " " << currentPoint << std::endl;

            if(i == 0)
                break;

            if(currentPoint == 0)
            {
                left = 9999;
                min = currentPoint;
            }
            else
            {
                left = energyMap.at<double>(i - 1, currentPoint - 1);
                min = currentPoint - 1;
            }
            
            center = energyMap.at<double>(i - 1, currentPoint);

            if (center < left)
                min = currentPoint;

            if (currentPoint >= numCols - 1)
                right = 99999;
            else
            {
                right = energyMap.at<double>(i - 1, currentPoint + 1);
            }
            
            if (right < center && right < left)
                min = currentPoint + 1;
            
            currentPoint = (int)min;

            energyMap.at<double>(i, currentPoint) = 999;

        }
                //std::cout << "here2" << std::endl;
        if(overHalf == 1)
            originalImage = originalImage.colRange(0, numCols - 1);
        else
            originalImage = originalImage.colRange(1, numCols);

    }
    else //direction == HORIZONTAL
    {
        cv::Point min_loc, max_loc;
        double min, max, down, center, up;
        Mat rightColumn = energyMap.col(numCols - 1);
        int overHalf = 0;
        cv::minMaxLoc(rightColumn, &min, &max, &min_loc, &max_loc);
        currentPoint = min_loc.y - iteration;
        if(currentPoint < 0)
            currentPoint = 0;
        if(currentPoint > numCols / 2)
            overHalf = 1;  
        removeSeam(originalImage, currentPoint, numCols - 1, direction, overHalf);
        energyMap.at<double>(currentPoint, numCols - 1) = 999;
        for(int j = numCols - 2; j > -1; j--)
        {
            removeSeam(originalImage, currentPoint, j, direction, overHalf);
            //originalImage.at<cv::Vec3b>(currentPoint, j)[0]=0; 
            //originalImage.at<cv::Vec3b>(currentPoint, j)[1]=0;
            //originalImage.at<cv::Vec3b>(currentPoint, j)[2]=0;
            if(j == 0)
                break;

            if(currentPoint == 0)
            {
                up = 99999;
                min = currentPoint;
            }
            else
            {
                up = energyMap.at<double>(currentPoint - 1, j - 1);
                min = currentPoint - 1;
            }
            center = energyMap.at<double>(currentPoint, j - 1);
            if (center < up)
                min = currentPoint;

            if (currentPoint >= numRows - 1)
                down = 99999;
            else
            {
                down = energyMap.at<double>(currentPoint + 1, j - 1);
            }

            if (down < center && down < up)
                min = currentPoint + 1;
            
            currentPoint = (int)min;

            energyMap.at<double>(currentPoint, j) = 999;
        }
        originalImage = originalImage.rowRange(0, numRows - 1);
    }

    //namedWindow("Seam", WINDOW_AUTOSIZE); imshow("Seam", originalImage);
    return originalImage;
}

void removeSeam(Mat input, int i, int j, int direction, int overHalf)
{
    int numRows, numCols;
    numRows = input.rows;
    numCols = input.cols;
    if(direction == VERTICAL)
    {
        if(overHalf == 1)  //shift pixels on left side of seam
        {
            Mat newRow;
            Mat firstHalf, secondHalf;
            Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
            if(j == 0)
            {
                secondHalf = input.rowRange(i, i + 1).colRange(j + 1, numCols);
                hconcat(secondHalf, dummy, newRow);
            }
            else if (j == numCols - 1)
            {
                firstHalf = input.rowRange(i, i + 1).colRange(0, j);
                hconcat(firstHalf, dummy, newRow);
            }
            else
            {
                firstHalf = input.rowRange(i, i + 1).colRange(0, j);
                secondHalf = input.rowRange(i, i + 1).colRange(j + 1, numCols);
                hconcat(secondHalf, dummy, secondHalf);
                hconcat(firstHalf, secondHalf, newRow);
            }
            newRow.copyTo(input.row(i));
        }
        else
        {
            Mat newRow;
            Mat firstHalf, secondHalf;
            Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
            if(j == 0)
            {
                secondHalf = input.rowRange(i, i + 1).colRange(j + 1, numCols);
                hconcat(dummy, secondHalf, newRow);
            }
            else if (j == numCols - 1)
            {
                firstHalf = input.rowRange(i, i + 1).colRange(0, j);
                hconcat(firstHalf, dummy, newRow);
            }
            else
            {
                firstHalf = input.rowRange(i, i + 1).colRange(0, j);
                secondHalf = input.rowRange(i, i + 1).colRange(j + 1, numCols);
                hconcat(dummy, firstHalf, firstHalf);
                hconcat(firstHalf, secondHalf, newRow);
            }
            newRow.copyTo(input.row(i));
        }
    }
    else //direction == HORIZONTAL
    {
        Mat newCol;
        Mat firstHalf, secondHalf;
        Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
        if(i == 0)
        {
            secondHalf = input.rowRange(i + 1, numRows).colRange(j, j + 1);
            transpose(secondHalf, secondHalf);
            hconcat(secondHalf, dummy, newCol);
        }
        else if (i == numRows - 1)
        {
            firstHalf = input.rowRange(0, i).colRange(j, j + 1);
            transpose(firstHalf, firstHalf);
            hconcat(firstHalf, dummy, newCol);
        }
        else
        {
            firstHalf = input.rowRange(0, i).colRange(j, j + 1);
            transpose(firstHalf, firstHalf);
            secondHalf = input.rowRange(i + 1, numRows).colRange(j, j + 1);
            transpose(secondHalf, secondHalf);
            hconcat(secondHalf, dummy, secondHalf);
            hconcat(firstHalf, secondHalf, newCol);
        }
        transpose(newCol, newCol);
        newCol.copyTo(input.col(j));
    }
}