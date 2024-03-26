#include <iostream>
#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <semaphore.h>
#include <sched.h>

using namespace cv;
using namespace std;

void end_delay_test(void);

sem_t logSem, stopSem;
struct timespec startTime, endTime;
bool stop_logging = false;

long long totalFrameTime_nsec = 0; // Total time for frames in nanoseconds
int frameCount = 0;                // Count of frames processed
int numFramesForAvg = 20;          // Number of frames to average over
long double avgExecTime_msec = 1.7;
int numFramesForAvg_init = 15;

FILE *fpt;

#define ESCAPE_KEY 27
#define FRAME_DELAY 33  // Frame delay in milliseconds (about 30 frames per second)
#define NSEC_PER_SEC (1000000000)
#define DELAY_TICKS (1)
#define ERROR (-1)
#define OK (0)

pthread_mutex_t cameraMutex; // Mutex for camera access

struct ThreadData {
    VideoCapture* cam;
};

int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
    // Ensure start and stop times are not NULL
    if (start == NULL || stop == NULL || delta_t == NULL) {
        return ERROR; // return error if any input is NULL
    }

    // Calculate the time difference
    delta_t->tv_sec = stop->tv_sec - start->tv_sec;
    delta_t->tv_nsec = stop->tv_nsec - start->tv_nsec;

    // Normalize the time so that tv_nsec is less than 1 second
    if (delta_t->tv_nsec < 0) {
        --delta_t->tv_sec;
        delta_t->tv_nsec += NSEC_PER_SEC;
    }

    // Ensure that the time difference is not negative
    if (delta_t->tv_sec < 0 || (delta_t->tv_sec == 0 && delta_t->tv_nsec < 0)) {
        // The stop time is before the start time, which should not happen
        delta_t->tv_sec = 0;
        delta_t->tv_nsec = 0;
        return ERROR; // return an error code
    }

    return OK; // return OK for successful execution
}

void* LoggingThread(void* threadp) {
    struct timespec ts;
    fprintf(fpt,"Frame no, Execution time, Jitter time\n");
    while (true) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += 1; // Wait for up to 1 second
        if (sem_timedwait(&stopSem, &ts) == -1) {
            if (errno != ETIMEDOUT) {
                break;
            }
        } else {
            break;
        }
        if (sem_trywait(&logSem) == 0) {
            struct timespec elapsedTime;
            delta_t(&endTime, &startTime, &elapsedTime);
            
            long long frameTime_nsec = elapsedTime.tv_sec * NSEC_PER_SEC + elapsedTime.tv_nsec;
            long double frameTime_msec = (long double)frameTime_nsec / (1000000);
            
            cout <<"Frame: " << frameCount << " Execution time: "<< frameTime_msec << " msec Jitter time:" << avgExecTime_msec - frameTime_msec << " msec" << endl;
            fprintf(fpt,"%d, %Lf, %Lf\n",frameCount, frameTime_msec, avgExecTime_msec - frameTime_msec);
            
            totalFrameTime_nsec += frameTime_nsec;
            frameCount++;

            // Calculate and display average framerate every numFramesForAvg frames
            if (frameCount == numFramesForAvg_init) {
                if (totalFrameTime_nsec > 0) {
                    long double avgTimePerFrame_msec = (long double)totalFrameTime_nsec / (numFramesForAvg * 1000000);
                    long double avgFramerate = 1000 / avgTimePerFrame_msec;
                    cout << endl << "Avergare Execution time: "<< avgTimePerFrame_msec <<"msec Average Framerate: " << avgFramerate << " fps (Calculated for " << numFramesForAvg_init << " Frames)" << endl << endl;
                    //Update avg excution time to calulate jitter
                    avgExecTime_msec = avgTimePerFrame_msec;
                    
                    // Use avg from fisrt 15 to calculate jitter
                    numFramesForAvg_init = numFramesForAvg;
                }
                
                // Reset counters
                totalFrameTime_nsec = 0;
                frameCount = 0;
            }
        }
    }
    return nullptr;
}

void* CannyThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    int PreiodMS = 100;
    cout << "********************Entered Canny Thread********************" <<endl;
    Mat frame, src_gray, detected_edges, dst;
    namedWindow("Edge Map", WINDOW_AUTOSIZE);

    int lowThreshold = 0;
    int max_lowThreshold = 100;
    int kernel_size = 3;
    createTrackbar("Min Threshold:", "Edge Map", &lowThreshold, max_lowThreshold);
    while(true) {
        pthread_mutex_lock(&cameraMutex);
        bool success = data->cam->read(frame);
        pthread_mutex_unlock(&cameraMutex);
        
        clock_gettime(CLOCK_REALTIME, &startTime);
        if (!success) {
            cerr << "Error: Could not grab a frame" << endl;
            break;
        }

        cvtColor(frame, src_gray, COLOR_BGR2GRAY);
        blur(src_gray, detected_edges, Size(3,3));
        Canny(detected_edges, detected_edges, lowThreshold, lowThreshold * 3, kernel_size);
        dst = Scalar::all(0);
        frame.copyTo(dst, detected_edges);
        imshow("Edge Map", dst);
        clock_gettime(CLOCK_REALTIME, &endTime);
        sem_post(&logSem);
        char c = (char)waitKey(PreiodMS);
        if (c == ESCAPE_KEY) {
            stop_logging = true;
            break;
        }
    }

    return nullptr;
}

void* HoughLinesThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    int PreiodMS = 60;
    cout << "********************Entered Hough Lines Thread********************" <<endl;
    Mat frame, dst, cdst, cdstP;
    namedWindow("Detected Lines", WINDOW_AUTOSIZE);
    
    while(true) {
        pthread_mutex_lock(&cameraMutex);
        bool success = data->cam->read(frame);
        pthread_mutex_unlock(&cameraMutex);

        if (!success) {
            cerr << "Error: Could not grab a frame" << endl;
            break;
        }
        
        clock_gettime(CLOCK_REALTIME, &startTime);
        // Convert to grayscale
        cvtColor(frame, frame, COLOR_BGR2GRAY);

        // Edge detection
        Canny(frame, dst, 80, 240, 3);
        
        // Copy edges to the images that will display the results in BGR
        cvtColor(dst, cdst, COLOR_GRAY2BGR);
        cdstP = cdst.clone();

        // Probabilistic Line Transform
        vector<Vec4i> linesP; // will hold the results of the detection
        HoughLinesP(dst, linesP, 1, CV_PI/180, 50, 50, 10); // runs the actual detection

        // Draw the lines
        for(size_t i = 0; i < linesP.size(); i++) {
            Vec4i l = linesP[i];
            line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, LINE_AA);
        }

        // Show results
        imshow("Detected Lines", cdstP);
        clock_gettime(CLOCK_REALTIME, &endTime);
        sem_post(&logSem);
        char c = (char)waitKey(PreiodMS);
        if (c == ESCAPE_KEY) {
            stop_logging = true;
            break;
        }
    }
    return nullptr;
}

void* PyrUpDownThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    int PreiodMS = 500;
    cout << "********************Entered Pyramid Up and Down Thread********************" <<endl;
    Mat frame;
    namedWindow("Pyramids Demo", WINDOW_AUTOSIZE);

    while(true) {
        pthread_mutex_lock(&cameraMutex);
        
        bool success = data->cam->read(frame);
        pthread_mutex_unlock(&cameraMutex);

        if (!success) {
            cerr << "Error: Could not grab a frame" << endl;
            break;
        }

        imshow("Pyramids Demo", frame);

        char c = (char)waitKey(0); // Wait for a key press
        clock_gettime(CLOCK_REALTIME, &startTime);
        if (c == ESCAPE_KEY) {
            break;
        } else if (c == 'i') {
            Mat temp;
            pyrUp(frame, temp, Size(frame.cols * 2, frame.rows * 2));
            frame = temp;
            printf("** Zoom In: Image x 2 \n");
        } else if (c == 'o') {
            Mat temp;
            pyrDown(frame, temp, Size(frame.cols / 2, frame.rows / 2));
            frame = temp;
            printf("** Zoom Out: Image / 2 \n");
        }

        imshow("Pyramids Demo", frame); // Redisplay the frame after zoom operation
        clock_gettime(CLOCK_REALTIME, &endTime);
        sem_post(&logSem);
        waitKey(PreiodMS); // Add delay to observe the zoom effect
    }
    return nullptr;
}

int setRealTimeScheduling(int priority, String sched_policy) {
    struct sched_param sch_params;
    sch_params.sched_priority = (sched_policy == "SCHED_FIFO") ? priority : 0;
    if(sched_policy == "SCHED_FIFO") {
      cout << endl << "Schedule Policy selected: SCHED_FIFO" << endl << endl;
      return sched_setscheduler(0, SCHED_FIFO, &sch_params);
    } else {
      cout << endl << "Schedule Policy selected: SCHED_OTHER" << endl << endl;
      return sched_setscheduler(0, SCHED_OTHER, &sch_params);
    }
}

int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv,
                             "{res_w w|640|camera resolution width}"
                             "{res_h h|480|camera resolution height}"
                             "{CT   |  |continuous transform type}"
                             "{sched_policy   |  |scheduling policy type}");

    VideoCapture cam0(0);
    if (!cam0.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }
    
    fpt = fopen("ContinuousTransform.csv", "w+");

    // Initialize the camera mutex
    pthread_mutex_init(&cameraMutex, NULL);
    sem_init(&logSem, 0, 0);
    
    cout << endl << "Excercise 4 question 5a:" << endl;

    int width = 640;
    width = parser.get<int>("res_w");
    int height = 480;
    height = parser.get<int>("res_h");
    
    cout << endl << "Resolution set to: " << width << "x"<< height << endl;

    cam0.set(CAP_PROP_FRAME_WIDTH, width);
    cam0.set(CAP_PROP_FRAME_HEIGHT, height);

    String cmd = parser.get<String>("CT");
    if (cmd.empty()) {
        cout << "No command provided. Please provide --CT=<command>." << endl;
        return -1;
    }
    
    cout << endl << "Transform selected: " << cmd << endl;
    
    String sched_policy = "SCHED_OTHER";
    sched_policy = parser.get<String>("sched_policy");
    
    if (setRealTimeScheduling(99, sched_policy) != 0) { // 99 is a high priority
        cerr << "Failed to set real-time scheduling policy." << endl;
        return -1;
    }

    ThreadData data;
    data.cam = &cam0;
    
    pthread_t logThread;
    pthread_create(&logThread, NULL, LoggingThread, NULL);

    pthread_t thread;
    int rc;
    cpu_set_t cpuset;

    // Initialize CPU set to the desired CPU(s)
    CPU_ZERO(&cpuset);   // Clear the CPU set
    CPU_SET(0, &cpuset); 

    if (cmd == "canny") {
        rc = pthread_create(&thread, NULL, CannyThread, &data);
    } else if (cmd == "houghline") {
        rc = pthread_create(&thread, NULL, HoughLinesThread, &data);
    } else if (cmd == "pyrUpDown") {
        rc = pthread_create(&thread, NULL, PyrUpDownThread, &data);
    } else {
        cout << "Invalid command provided: " << cmd << endl;
        return -1;
    }
    
    if (rc == 0) {
            pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
            pthread_setaffinity_np(logThread, sizeof(cpu_set_t), &cpuset);
        }

    if (rc) {
        cout << "Error: unable to create thread," << rc << endl;
        return -1;
    }

    pthread_join(thread, NULL); // Wait for the thread to finish
    sem_post(&stopSem);
    pthread_join(logThread, NULL);

    pthread_mutex_destroy(&cameraMutex);
    sem_destroy(&logSem);
    fclose(fpt);

    return 0;
}
