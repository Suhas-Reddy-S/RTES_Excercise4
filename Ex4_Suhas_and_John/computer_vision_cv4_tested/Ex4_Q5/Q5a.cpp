#include <iostream>
#include <pthread.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define ESCAPE_KEY 27
#define FRAME_DELAY 33  // Frame delay in milliseconds (about 30 frames per second)

pthread_mutex_t cameraMutex; // Mutex for camera access

struct ThreadData {
    VideoCapture* cam;
};

void* CannyThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
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

        char c = (char)waitKey(FRAME_DELAY);
        if (c == ESCAPE_KEY) {
            break;
        }
    }

    return nullptr;
}

void* HoughLinesThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    Mat frame, dst, cdst, cdstP;
    namedWindow("Detected Lines", WINDOW_AUTOSIZE);

    while (true) {
        pthread_mutex_lock(&cameraMutex);
        bool success = data->cam->read(frame);
        pthread_mutex_unlock(&cameraMutex);

        if (!success) {
            cerr << "Error: Could not grab a frame" << endl;
            break;
        }

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

        char c = (char)waitKey(1);
        if (c == ESCAPE_KEY) {
            break;
        }
    }
    return nullptr;
}

void* PyrUpDownThread(void* arg) {
    ThreadData* data = static_cast<ThreadData*>(arg);
    Mat frame;
    namedWindow("Pyramids Demo", WINDOW_AUTOSIZE);
    int delayMs = 500; // 500 milliseconds delay

    while (true) {
        pthread_mutex_lock(&cameraMutex);
        bool success = data->cam->read(frame);
        pthread_mutex_unlock(&cameraMutex);

        if (!success) {
            cerr << "Error: Could not grab a frame" << endl;
            break;
        }

        imshow("Pyramids Demo", frame);

        char c = (char)waitKey(0); // Wait for a key press
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
        waitKey(delayMs); // Add delay to observe the zoom effect
    }
    return nullptr;
}

int main(int argc, char *argv[]) {
    CommandLineParser parser(argc, argv,
                             "{res_w w|640|camera resolution width}"
                             "{res_h h|480|camera resolution height}"
                             "{CT   |  |continuous transform type}");

    VideoCapture cam0(0);
    if (!cam0.isOpened()) {
        cerr << "Error: Could not open camera" << endl;
        return -1;
    }

    // Initialize the camera mutex
    pthread_mutex_init(&cameraMutex, NULL);

    int width = parser.get<int>("res_w");
    int height = parser.get<int>("res_h");

    cam0.set(CAP_PROP_FRAME_WIDTH, width);
    cam0.set(CAP_PROP_FRAME_HEIGHT, height);

    String cmd = parser.get<String>("CT");
    if (cmd.empty()) {
        cout << "No command provided. Please provide --CT=<command>." << endl;
        return -1;
    }

    ThreadData data;
    data.cam = &cam0;

    pthread_t thread;
    int rc;

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

    if (rc) {
        cout << "Error: unable to create thread," << rc << endl;
        return -1;
    }

    pthread_join(thread, NULL); // Wait for the thread to finish

    pthread_mutex_destroy(&cameraMutex);

    return 0;
}

