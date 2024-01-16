#include <fstream>
#include <iostream>
#include <vector>

#include "conv_0.layer.hpp"

void read(std::string path, std::vector<unsigned char> &buffer) {
  std::ifstream read_file(path, std::ios::binary);
  if (!read_file) {
    std::cout << "无法打开二进制文件" << std::endl;
    return;
  }
  read_file.seekg(0, std::ios::end);
  std::streampos size = read_file.tellg();
  read_file.seekg(0, std::ios::beg);
  buffer.resize(size);
  read_file.read(reinterpret_cast<char *>(buffer.data()), size);
  read_file.close();
}

void meanStd(const std::vector<unsigned char> &input, std::vector<float> &output) {
  float scale = 1.0f / 255.f;
  float bias = -0.5f;
  for (int i = 0; i < 3; i++) {  // bgr 顺序
    for (int j = 0; j < input.size() / 3; j++) {
      int d = input[i + 3 * j];
      float value = float(d) * scale + bias;
      output.push_back(value);
    }
  }
}

void conv_0(std::vector<float> &input, const std::vector<int> &input_dims, std::vector<float> &output,
            std::vector<int> &output_dims) {
  const int pad = 1;
  const int stride = 2;
  const int kernel_shape = 3;
  NaiveConv(input.data(), output.data(), conv_0_w_1.data(), conv_0_b.data(), input_dims, output_dims, stride, stride,
            kernel_shape, kernel_shape, pad, pad);
}

void save(std::string fname, const std::vector<float> &output) {
  FILE *fp = fopen(fname.c_str(), "wb");
  if (!fp) {
    std::cout << "无法创建文件：" << fname << std::endl;
    return;
  }
  for (int n = 0; n < output.size(); ++n) {
    fprintf(fp, "%.9f\n", output[n]);
  }
  fclose(fp);
}

int main() {
  std::vector<unsigned char> data;
  read("examples/native/input.1.3.224.224.bin", data);
  std::cout << data.size() << std::endl;

  std::vector<float> input;
  meanStd(data, input);
  save("build/0-mean-std.txt", input);
  std::vector<int> input_dims = {1, 3, 224, 224};

  std::vector<float> output(1*56*112*112);
  std::vector<int> output_dims = {1, 56, 112, 112};
  conv_0(input, input_dims, output, output_dims);
  save("build/0-output.txt", output);

  std::vector<std::vector<float>> ker = {
      {-0.0074548558332026005, 0.010620209388434887, -0.007965154014527798},
      {-0.002728730672970414, 0.08870813995599747, 0.0979325920343399},
      {-0.015970367938280106, -0.0011951475171372294, 0.015254144556820393},
  };

  float d =
      ker[1][1] * input[0] + ker[1][2] * input[1] + ker[2][1] * input[224] + ker[2][2] * input[225] + 0.272976279258728;
  std::cout << d << std::endl;

  return 0;
}
