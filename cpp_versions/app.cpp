#include <vector>
#include <cmath>
#include <random>
#include <iostream>
// be careful with macros. 
#define PI 3.14159265f
#define E_VAL 2.71828182f

using namespace std;

// point defined by x,y floats - 16 bytes total
typedef struct {
    float x; //4 bytes
    float y; //4 bytes
    float z; //4 bytes
    int _id; //4 bytes  
} point;

/*same thing as the point 
just want the names to be intuitive*/ 
typedef struct{
    float dx;
    float dy;
    float dz;
    int _id;
} gradient;

void randomize_points(point* point_arr, int itts){
    //Random number generator -- can probably do this in the header honestly. 
    // static keeps these 2 things alive between function calls!
    static random_device rd; // create a random device - non-deterministic seed
    static mt19937 engine(rd()); // seed mersenne twister engine
    uniform_real_distribution<float> dis(-5.f, 5.f);

    for (int i =0 ; i<itts; i++){
        point_arr[i].x = dis(engine);
        point_arr[i].y = dis(engine);
    }
}

void print_points(point* point_arr, int itts){
    cout<<"Hey nice point list\nX,Y,Z\n";
    for (int i=0; i<itts; i++){
        cout<<point_arr[i].x<<","
        <<point_arr[i].y<<","
        <<point_arr[i].z<<"\n";
    }
}

// Start explicitly with floats
void Ackley_f(point* point_arr, int itts) {
    float term1;
    float term2;

    for (int i = 0; i < itts; i++){
        // Using expf, sqrtf, and cosf for 32-bit precision
        term1 = -20.0f * expf(-0.2f * sqrtf(0.5f * (point_arr[i].x*point_arr[i].x + point_arr[i].y*point_arr[i].y)));
        term2 = -expf(0.5f * (cosf(2.0f * PI * point_arr[i].x) + cosf(2.0f * PI * point_arr[i].y)));
        point_arr[i].z = term1 + term2 + E_VAL + 20.0f;
    }
}

int main(){
    const int n_pts = 1024;
    point Points[n_pts] = {0.0f16}; // single point list. we modify in place here. dont be a coward.

    randomize_points(&Points[0], n_pts);
    Ackley_f(&Points[0], n_pts);

    print_points(&Points[0], n_pts);

    return 0;
}