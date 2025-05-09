#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/triangulated_grid.h>
#include <igl/per_vertex_normals.h>
#include <igl/barycentric_interpolation.h>
#include <igl/blue_noise.h>
#include <igl/doublearea.h>
#include <igl/PI.h>
#include <Eigen/Core>
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <cstdlib>
#include <filesystem>

const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", "\n");

void sample_pc_with_blue_noise(
  const int & num_points,
  const Eigen::MatrixXd & mesh_v,
  const Eigen::MatrixXi & mesh_f,
  Eigen::MatrixXd & pc,
  Eigen::MatrixXd & normals)
{
  if(mesh_v.size() == 0 || mesh_f.size() == 0) {
    std::cerr << "Error: Input mesh is empty.\n";
    return;
  }

  Eigen::VectorXd A;
  igl::doublearea(mesh_v, mesh_f, A);
  double radius = sqrt(((A.sum()*0.5/(num_points*0.6162910373))/igl::PI));
  std::cout << "Initial Blue noise radius: " << radius << "\n";
  
  Eigen::MatrixXd B;
  Eigen::VectorXi I;
  int max_attempts = 1; 
  for(int attempt = 0; attempt < max_attempts; ++attempt)
  {
    igl::blue_noise(mesh_v, mesh_f, radius, B, I, pc);
    std::cout<<"successfully generate a set of blue noise!"<<std::endl;
    if(pc.rows() >= num_points * 0.9 && pc.rows() <= num_points * 1.1)
    {
      break;
    }
    if(pc.rows() == 0) {
      std::cerr << "Error: Blue noise sampling generated an empty point cloud. Attempt: " << attempt + 1 << "\n";
      break;
    }
    radius *= sqrt(num_points * 1.0 / pc.rows());
    //std::cout << "Adjusted Blue noise radius: " << radius << " (Attempt " << attempt + 1 << ")\n";
  }

  if(pc.rows() == 0) {
    std::cerr << "Error: Blue noise sampling failed to generate points.\n";
    return;
  }

  if (pc.rows() > num_points)
  {
    std::cout << "Trimming point cloud from " << pc.rows() << " to " << num_points << " points\n";
    std::vector<int> indices(pc.rows());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine{});
    indices.resize(num_points);
    Eigen::MatrixXd trimmed_pc(num_points, pc.cols());
    //Eigen::MatrixXd trimmed_normals(num_points, normals.cols());
    std::cout << "pc.rows(): " << pc.rows() << ", pc.cols(): " << pc.cols() << std::endl;
    //std::cout << "normals.rows(): " << normals.rows() << ", normals.cols(): " << normals.cols() << std::endl;    
    for (int i = 0; i < num_points; ++i)
    {
      //std::cout <<"for trimming" << " i is" << i << std::endl;
      trimmed_pc.row(i) = pc.row(indices[i]);
      //std::cout<<"1"<<std::endl;
      //trimmed_normals.row(i) = normals.row(indices[i]);
    }
    pc = trimmed_pc;
    //normals = trimmed_normals;
  }
  else if (pc.rows() < num_points)
  {
    std::cout << "Padding point cloud from " << pc.rows() << " to " << num_points << " points\n";
    Eigen::MatrixXd padded_pc(num_points, pc.cols());
    //Eigen::MatrixXd padded_normals(num_points, normals.cols());
    //std::cout << "pc.rows(): " << pc.rows() << ", pc.cols(): " << pc.cols() << std::endl;
    //std::cout << "normals.rows(): " << normals.rows() << ", normals.cols(): " << normals.cols() << std::endl;    
    for (int i = 0; i < num_points; ++i)
    {
      //std::cout <<"for padding" << " i is"<< i << std::endl;
      padded_pc.row(i) = pc.row(i % pc.rows());
      //std::cout<<"2"<<std::endl;
      //padded_normals.row(i) = normals.row(i % normals.rows());
    }
    pc = padded_pc;
    //normals = padded_normals;
  }

  if(pc.rows() != num_points)
  {
    std::cerr << "Warning: Generated " << pc.rows() << " points instead of " << num_points << "\n";
  }

  Eigen::MatrixXd vertex_normals;
  igl::per_vertex_normals(mesh_v, mesh_f, vertex_normals);
  igl::barycentric_interpolation(vertex_normals, mesh_f, B, I, normals);
  normals.rowwise().normalize();
}

void write_matrix_to_csv(
  const std::string & filename,
  const Eigen::MatrixXd & T)
{
  std::ofstream file(filename.c_str());
  if(!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filename << "\n";
    return;
  }
  file << T.format(CSVFormat);
}

int main(int argc, char *argv[])
{
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " <mesh_file_partA.obj> <mesh_file_partB.obj> <save_root_directory>\n";
    return 1;
  }

  std::string mesh_file_left = argv[1];
  std::string mesh_file_right = argv[2];
  std::string save_root_directory = argv[3];
  Eigen::MatrixXd L_mesh_v, R_mesh_v;
  Eigen::MatrixXi L_mesh_f, R_mesh_f;

  if (!igl::read_triangle_mesh(mesh_file_left, L_mesh_v, L_mesh_f))
  {
    std::cerr << "Failed to load mesh from " << mesh_file_left << std::endl;
    return 1;
  }

  if (!igl::read_triangle_mesh(mesh_file_right, R_mesh_v, R_mesh_f))
  {
    std::cerr << "Failed to load mesh from " << mesh_file_right << std::endl;
    return 1;
  }

  std::cout << "Sampling point clouds...\n";
  int num_points = 1024;
  Eigen::MatrixXd L_pc, R_pc;
  Eigen::MatrixXd L_normal, R_normal;
  sample_pc_with_blue_noise(num_points, L_mesh_v, L_mesh_f, L_pc, L_normal);
  sample_pc_with_blue_noise(num_points, R_mesh_v, R_mesh_f, R_pc, R_normal);

  std::cout << "The target root file is " << save_root_directory << std::endl;

  std::cout << "Saving point clouds...\n";
  std::string L_pc_file = save_root_directory + '/' + "partA-pc.csv";
  std::string R_pc_file = save_root_directory + '/' + "partB-pc.csv";
  write_matrix_to_csv(L_pc_file, L_pc);
  write_matrix_to_csv(R_pc_file, R_pc);

  std::cout << "Saving normals...\n";
  std::string L_normal_file = save_root_directory + '/' + "partA-normal.csv";
  std::string R_normal_file = save_root_directory + '/' + "partB-normal.csv";
  write_matrix_to_csv(L_normal_file, L_normal);
  write_matrix_to_csv(R_normal_file, R_normal);

  std::cout << "Saved point clouds and normals to " << save_root_directory << std::endl;

  return 0;
}