#include <stdio.h>
#include <math.h>

// 定义一个结构体表示二维向量
typedef struct {
    float x;
    float y;
} Vec2;

// 定义一个结构体表示圆形
typedef struct {
    Vec2 center; // 圆心坐标
    float radius; // 半径
} Circle;

// 计算两个向量的距离
float distance(Vec2 a, Vec2 b) {
    return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

// 判断一个点是否在圆内
int inside(Circle c, Vec2 p) {
    return distance(c.center, p) <= c.radius;
}

// 定义画布的宽度和高度
#define WIDTH 80
#define HEIGHT 40

// 定义笑脸的半径、眼睛和嘴巴的位置和大小
#define FACE_RADIUS 15.0f
#define EYE_RADIUS 1.5f
#define EYE_OFFSET_X 5.0f
#define EYE_OFFSET_Y 5.0f
#define MOUTH_RADIUS 7.0f

int main() {
    // 创建一个字符数组表示画布，每个字符对应一个像素点，初始值为空格
    char canvas[HEIGHT][WIDTH + 1];
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            canvas[i][j] = ' ';
        }
        canvas[i][WIDTH] = '\0'; // 每行最后一个字符为字符串结束符
    }

    // 创建一个圆形表示笑脸，圆心在画布中心，半径为FACE_RADIUS
    Circle face = {{WIDTH / 2.0f, HEIGHT / 2.0f}, FACE_RADIUS};

    // 创建两个圆形表示眼睛，圆心在笑脸上方左右偏移EYE_OFFSET_X，上下偏移EYE_OFFSET_Y，半径为EYE_RADIUS 
    Circle left_eye = {{face.center.x - EYE_OFFSET_X, face.center.y + EYE_OFFSET_Y}, EYE_RADIUS};
    Circle right_eye = {{face.center.x + EYE_OFFSET_X, face.center.y + EYE_OFFSET_Y}, EYE_RADIUS};

    // 创建一个圆形表示嘴巴，圆心在笑脸下方中间偏移MOUTH_RADIUS / 2，半径为MOUTH_RADIUS 
    Circle mouth = {{face.center.x, face.center.y - MOUTH_RADIUS / 2}, MOUTH_RADIUS};

    // 遍历画布上的每个像素点，判断是否在笑脸、眼睛或嘴巴内部，如果是则用*号填充，否则保持空格 
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            Vec2 p = {j + 0.5f, i + 0.5f}; // 像素点的中心坐标 
            if (inside(face, p)) { // 在笑脸内部 
                if (inside(left_eye, p) || inside(right_eye, p)) { // 在眼睛内部 
                    canvas[i][j] = '*';
                } else if (inside(mouth, p)) { // 在嘴巴内部 
                    if ((p.x - mouth.center.x) * (p.x - mouth.center.x) > MOUTH_RADIUS * MOUTH_RADIUS / 4) { // 在嘴巴边缘 
                        canvas[i][j] = '*';
                    }
                } else { // 在笑脸外部 
                    if (distance(face.center, p) > FACE_RADIUS - 0.5f) { // 在笑脸边缘 
                        canvas[i][j] = '*';
                    }
                }
            }
        }
    }

    // 打印画布上的字符，输出笑脸图案 
    for (int i = 0; i < HEIGHT; i++) {
        printf("%s\n", canvas[i]);
    }

    return 0;
}