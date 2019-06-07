
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
//#include "/usr/local/Cellar/opencv/4.1.0_2/include/opencv4/opencv2/opencv.hpp"
#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
//#include <emscripten.h>


#define HORIZONTAL 0
#define VERTICAL 1

#define MULTITHREAD 1

using namespace cv;


typedef struct thread_data_s {
    int tid;
    int num_threads;
    int num_elements;
    int *seam;
    Mat* image;
    int offset;
    int chunk_size;
    double *partial_sum;
} thread_data_t;

Mat calculateImageGradient(Mat input);
Mat calculateImageEnergy(Mat input, int direction);
Mat findSeam(Mat energyMap, Mat originalImage, int direction, int iteration);
void removeSeam(Mat input, int i, int j, int location, int overHalf);
void *removeSeamMT(void* args);

int main(int argc, const char * argv[]) {
    clock_t start = clock();
    int dir;
    if(strcmp (argv[2], "HORIZONTAL") == 0)
	dir = HORIZONTAL;
    else if(strcmp (argv[2], "VERTICAL") == 0)
	dir = VERTICAL;
    else
    {
	    std::cout << "Sorry unknown direction, please type either 'HORIZONTAL' or 'VERTICAL' " << std::endl;
	    return 0;
    }
    //char* file = argv[1];
    int numSeams = atoi(argv[3]);
    std::cout << argv[1] << std::endl;
    Mat testImage, outputImage, outputImage2;
    testImage = cv::imread(argv[1], IMREAD_COLOR);
    namedWindow("Display Window", WINDOW_AUTOSIZE);
    //imshow("Display Window", outputImage2);
    for(int i = 0; i < numSeams; i++)
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
    namedWindow("Resized Image", WINDOW_AUTOSIZE); imshow("Resized Image", testImage);
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
    //namedWindow("Cumulative Energy Map", WINDOW_AUTOSIZE); imshow("Cumulative Energy Map", colorMap);

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
        int *seam = (int *) malloc (sizeof (int) * numRows);
        Mat bottomRow = energyMap.row(numRows - 1);
        int overHalf = 0;
        cv::minMaxLoc(bottomRow, &min, &max, &min_loc, &max_loc);
        currentPoint = min_loc.x - iteration;
        if(currentPoint < 0)
            currentPoint = 0;
        if(currentPoint > numCols / 2)
            overHalf = 1;
        //std::cout << currentPoint << std::endl;
        //removeSeam(originalImage, numRows - 1, currentPoint, direction, overHalf);
        int seamIndex = 0;
        energyMap.at<double>(numRows - 1, currentPoint) = 999;
        seam [seamIndex] = currentPoint;
        seamIndex++;

        for(int i = numRows - 2; i > -1; i--)
        {
            if(MULTITHREAD == 0)
                removeSeam(originalImage, i, currentPoint, direction, overHalf);

            //originalImage.at<cv::Vec3b>(i, currentPoint)[0]=0; 
            //originalImage.at<cv::Vec3b>(i, currentPoint)[1]=0;
            //originalImage.at<cv::Vec3b>(i, currentPoint)[2]=0;
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

            seam [seamIndex] = currentPoint;
            seamIndex++;


        }
                //std::cout << "here2" << std::endl;

        if(MULTITHREAD == 0)
        {
            if(overHalf == 1)
                originalImage = originalImage.colRange(0, numCols - 1);
            else
                originalImage = originalImage.colRange(1, numCols);
        }
        else    
        {
            int num_threads = 2;
            pthread_t *thread_id = (pthread_t *) malloc (num_threads * sizeof (pthread_t)); /* Data structure to store the thread IDs. */
            pthread_attr_t attributes; /* Thread attributes. */
            pthread_attr_init (&attributes); /* Initialize the thread attributes to the default values. */
            thread_data_t *thread_data = (thread_data_t *) malloc (sizeof (thread_data_t) * num_threads);
            int chunk_size = numRows / num_threads;

            for(int i = 0; i < num_threads; i++)
            {
                thread_data[i].tid = i; 
                thread_data[i].num_threads = num_threads;
                thread_data[i].seam = seam; 
                thread_data[i].image = &originalImage;
                thread_data[i].offset = i * chunk_size; 
                thread_data[i].chunk_size = chunk_size;
            }
            for (int i = 0; i < num_threads; i++)
                pthread_create (&thread_id[i], &attributes, removeSeamMT, (void *) &thread_data[i]);

            for (int i = 0; i < num_threads; i++)
                pthread_join (thread_id[i], NULL);

            free ((void *) seam);
            free ((void *) thread_id);
            originalImage = originalImage.colRange(0, numCols - 1);


        }

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
           // originalImage.at<cv::Vec3b>(currentPoint, j)[0]=0; 
           // originalImage.at<cv::Vec3b>(currentPoint, j)[1]=0;
           // originalImage.at<cv::Vec3b>(currentPoint, j)[2]=0;
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

void *removeSeamMT(void* args)
{
    thread_data_t *thread_data = (thread_data_t *) args;
    Mat* image = thread_data->image;
    Mat input = *image;
    int* seam = thread_data->seam;
    int offset = thread_data->offset;
    int chunk_size = thread_data->chunk_size;
    int seamIndex = offset;
    int numRows = input.rows;
    int numCols = input.cols;
    //for(int i = offset; i < offset + chunk_size; i++)
    for(int i = offset + chunk_size - 1; i >= offset; i--)
    {
        Mat newRow;
        Mat firstHalf, secondHalf;
        Mat dummy(1, 1, CV_8UC3, Vec3b(0, 0, 0));
        int j = seam[seamIndex];
        seamIndex++;
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
    pthread_exit (NULL);
}