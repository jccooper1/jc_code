#include <stdio.h>
#include <math.h>

// 定义一个结构体表示3维空间中的点
typedef struct {
  double x;
  double y;
  double z;
} Point;

// 定义一个结构体表示3维空间中的向量
typedef struct {
  double x;
  double y;
  double z;
} Vector;

// 定义一个函数计算两个点之间的距离
double distance(Point p1, Point p2) {
  return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2) + pow(p1.z - p2.z, 2));
}

// 定义一个函数计算两个向量的叉积
Vector cross_product(Vector v1, Vector v2) {
  Vector v;
  v.x = v1.y * v2.z - v1.z * v2.y;
  v.y = v1.z * v2.x - v1.x * v2.z;
  v.z = v1.x * v2.y - v1.y * v2.x;
  return v;
}

// 定义一个函数计算向量的模长
double norm(Vector v) {
  return sqrt(pow(v.x, 2) + pow(v.y, 2) + pow(v.z, 2));
}

// 定义一个函数计算3维空间中点到直线的距离
// 直线由过直线的一点p0和直线的方向向量v确定
double point_to_line(Point p, Point p0, Vector v) {
  // 计算点p到点p0的向量
  Vector u;
  u.x = p.x - p0.x;
  u.y = p.y - p0.y;
  u.z = p.z - p0.z;
  // 计算向量u和向量v的叉积
  Vector w = cross_product(u, v);
  // 计算叉积的模长除以向量v的模长，得到点到直线的距离
  return norm(w) / norm(v);
}

// 测试代码
int main() {
  // 定义一个点p
  Point p = {6, -2, 1};
  // 定义一个直线，过点p0，方向向量为v
  Point p0 = {1, 2, 3};
  Vector v = {1, 1, 1};
  // 计算点p到直线的距离
  double d = point_to_line(p, p0, v);
  // 打印结果
  printf("The distance from point (%.3f, %.3f, %.3f) to line is %.3f\n", p.x, p.y, p.z, d);
  return 0;
}
