#include<iostream>
#include<queue>
#include<windows.h>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;


void GetDesktopResolution(int& horizontal, int& vertical)
{
    RECT desktop;
    // Get a handle to the desktop window
    const HWND hDesktop = GetDesktopWindow();
    // Get the size of screen to the variable desktop
    GetWindowRect(hDesktop, &desktop);
    // The top left corner will have coordinates (0,0)
    // and the bottom right corner will have coordinates
    // (horizontal, vertical)
    horizontal = desktop.right;
    vertical = desktop.bottom;
}



//smoothen function part -1
int updateavg(int oldavg, int newval, int sz, int oldval)
{
    int newavg = ((oldavg * sz) - oldval + newval) / sz;
    return newavg;
}

//function to smoothen (to adjust sensitivity change the size), sliding window type -1
pair<int, int> smooth(queue<pair<int, int>>& wind, pair<int, int> curr, pair<int, int>& avg)
{
    if (wind.size() < 1)
    {
        wind.push(curr);
        return curr;
    }
    int x = updateavg(avg.first, curr.first, wind.size(), wind.front().first);
    int y = updateavg(avg.second, curr.second, wind.size(), wind.front().second);
    avg = { x,y };
    if (wind.size() > 1) wind.pop();
    wind.push(curr);
    return avg;
}

bool camcheck(int port)
{
    VideoCapture cam(port);
    return cam.isOpened();
}

int main() 
{
    queue<pair<int, int>> wind; //-1
    Mat img;//Mat object for images
    int up_width = 1280, up_height = 720;
    GetDesktopResolution(up_width, up_height);
    int alphx = up_width * 0.55, alphy = up_height * 0.35;
    Mat blank(alphy, alphx, CV_8UC3, Scalar(255, 255, 255));
    int startx = (up_width - alphx) / 2, starty = (up_height - alphy) / 2;
    namedWindow("BLANK");
    namedWindow("FACE");
    int hi = alphy / 3, wid = alphx / 9;
    for (int i = 0; i < 3; i++)
    {
        int vary = (hi * i) + (hi / 2);
        for (int j = 0; j < 9; j++)
        {
            int varx = (j * wid) + (wid / 2);
            char var = 'A' + (9 * i) + j;
            string t = "";
            t.push_back(var);
            if ((9 * i) + j == 26) t = ".";//for spacebar
            putText(blank, t, Point(varx, vary), FONT_HERSHEY_DUPLEX, 1.5, Scalar(0, 0, 0), 2);
        }
    }
    imshow("BLANK", blank);
    moveWindow("BLANK", startx, starty);
    int port = -1;
    for (int i = 0; i < 5; i++)//checking for 5 ports
    {
        if (camcheck(i))
        {
            port = i;
            break;
        }
    }
    if (port == -1)
    { //Error message if video source is not found
        cout << "Couldn't load video from the source.Make sure your camera is working properly." << endl;
        system("pause");
        return 0;
    }
    VideoCapture cap(port);
    CascadeClassifier face_cascade, eyes_cascade;//declaring a CascadeClassifier object
    face_cascade.load("./haarcascades/haarcascade_frontalface_alt.xml");//loading the cascade classifier//
    eyes_cascade.load("./haarcascades/haarcascade_eye.xml");
    pair<int, int> avg;
    char ans = ' ', prevchar = '1';
    int cnt = 0;
    while (true)
    {
        cap.read(img);//loading video frames from source to our matrix named frame, this returns bool and places each frame in img here
        resize(img, img, Size(up_width, up_height), INTER_LINEAR);
        Mat imcop;
        flip(img, img, 1);
        cvtColor(img, imcop, COLOR_BGR2GRAY);//converting to gray
        vector <Rect> faces;//Declaring a vector named faces
        face_cascade.detectMultiScale(imcop, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));//detecting the face

        for (auto i = 0; i < faces.size(); i++)
        { //for locating the face
            rectangle(img, Point(faces[i].x, faces[i].y), Point(faces[i].width + faces[i].x, faces[i].height + faces[i].y), Scalar(0, 0, 255), 4, 8, 0);//draw an ellipse on the face//
            Mat faceROI = imcop(faces[i]);//Taking area of the face as Region of Interest for eyes
            vector <Rect> eyes;//declaring a vector named eyes
            //HEAD IS 5 EYES WIDE
            eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(5, 5));//detect eyes in every face
            pair<int, int> left = { 0,0 }, right = { 0,0 }, midpt = { 0,0 };
            int dist = faces[i].width / 5;
            if (eyes.size() == 0)
            {
                continue;
            }
            int radius = 30;
            for (auto j = 0; j < eyes.size(); j++)
            { //for locating eyes
                if ((eyes[j].y) * 2 > faces[i].height) continue; //removing false positive nose detection
                if ((eyes[j].x + eyes[j].x + eyes[j].width) > (2 * dist) && (eyes[j].x + eyes[j].x + eyes[j].width) < (4 * dist))
                {
                    left = { faces[i].x + eyes[j].x + eyes[j].width * 0.5,faces[i].y + eyes[j].y + eyes[j].height * 0.5 };
                    radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                }
                else if ((eyes[j].x + eyes[j].x + eyes[j].width) > (6 * dist) && (eyes[j].x + eyes[j].x + eyes[j].width) < (8 * dist))
                {
                    right = { faces[i].x + eyes[j].x + eyes[j].width * 0.5, faces[i].y + eyes[j].y + eyes[j].height * 0.5 };
                    radius = cvRound((eyes[j].width + eyes[j].height) * 0.25);
                }
            }
            if (!left.first || !left.second || !right.first || !right.second)
            {
                continue;
            }
            midpt = { (left.first + right.first) / 2 ,(left.second + right.second) / 2 };
            circle(img, Point(left.first, left.second), 5, Scalar(255, 255, 0), FILLED);
            circle(img, Point(right.first, right.second), 5, Scalar(255, 255, 0), FILLED);
            if (wind.empty()) //-1
            {
                avg = midpt;
                wind.push(midpt);
            }
            else
            {
                midpt = smooth(wind, midpt, avg);
            }
            SetCursorPos(midpt.first, midpt.second);
            int refx = midpt.first - startx, refy = midpt.second - starty;
            for (int i = 0; i <= 3; i++)
            {
                if (hi * i > refy)
                {
                    if (i == 0)
                    {
                        cnt = 0;
                    }
                    else
                    {
                        for (int j = 0; j <= 9; j++)
                        {
                            if (wid * j > refx)
                            {
                                if (j == 0)
                                {
                                    cnt = 0;
                                }
                                else
                                {
                                    ans = 'A' + (9 * (i - 1)) + (j - 1);
                                    if ((9 * (i - 1)) + (j - 1) == 26) ans = ' ';
                                }
                                break;
                            }
                            else if (j == 9) cnt = 0;
                        }
                    }
                    break;
                }
                else if (i == 3) cnt = 0;
            }
            if (cnt == 0)
            {
                prevchar = ans;
                cnt++;
            }
            else if (prevchar != ans)
            {
                cnt = 0;
            }
            else cnt++;
            if (cnt == 8)
            {
                cout << ans;
                cnt = 0;
            }

        }
        resize(img, img, Size(320, 200), INTER_LINEAR);
        imshow("FACE", img);
        moveWindow("FACE", 50, 50);
        if ((char)waitKey(30) == (char)27) { //wait time for each frame is 30 milliseconds
            break;
        }
    }
    return 0;
}
