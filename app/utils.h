#ifndef CALIB_UTILS_H
#define CALIB_UTILS_H

#include <Eigen/Dense>
#include <fstream>
#include <iostream>



// From Raphael's: https://github.com/rFalque/utils/


// from: https://stackoverflow.com/a/39146048/2562693
// e.g., Eigen::MatrixXd matrix = read_csv<Eigen::MatrixXd>("csv_path.csv");
inline Eigen::MatrixXd readCSV(const std::string& path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  int rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  int cols = values.size() / rows;
  Eigen::MatrixXd out(rows, cols);
  for (int i = 0; i < values.size(); ++i) {
    int r = i / cols;
    int c = i % cols;
    out(r,c) = values[i];
  }
  indata.close();
  return out;

};



inline void writeCSV(std::string name, Eigen::MatrixXd matrix) {
  const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision,
                                         Eigen::DontAlignCols, ", ", "\n");
  std::ofstream file(name.c_str());
  file << matrix.format(CSVFormat);
  file.close();
}













#endif
