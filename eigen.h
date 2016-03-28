//
// Created by Debasis Dwivedy on 3/24/16.
//
#include <vector>
#ifndef CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H
#define CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H

#endif //CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H

#include "CImg.h"
using namespace std;
using namespace cimg_library;
CImg <float> *grey_scale(CImg <float> &image);
typedef struct {
  int food;
  int height;
  int width;
} eigen_database_table;


typedef struct {
  int index;
  int name;
} img_information;


void mean_normalized_image( CImg<float>& image,
                    CImg<float>& mean,
                    int size_of_trainingset ){
  for( int i = 0; i < image.width(); i++ ){
    for( int j = 0; j < image.height(); j++ ){
      *mean.data( i, j ) += *image.data( i, j ) / (size_of_trainingset);
    }
  }
}

void image_normalization( CImg<float>& image,
                      int index_position,
                      CImg<float>& mean,
                      float **&normalized_image ){
  for( int i = 0, k = 0; i < image.width(); i++ ){
    for( int j = 0; j < image.height(); j++, k++ ){
      normalized_image[index_position][k] = *image.data( i, j ) - *mean.data( i, j );
    }
  }
}

void create_covariance_matrix( float **& normalized_image,
                       float **& covariance_image_matrix,
                       int size_of_vector,
                       int size_of_trainingset ){
  float **image_transpose;
  int i, j, m;
  image_transpose = new float * [size_of_vector];
  for( i = 0; i < size_of_vector; i++ ){
    image_transpose[i] = new float [size_of_trainingset];
  }
  for( i = 0; i < size_of_vector; i++ ){
    for( j = 0; j < size_of_trainingset; j++ ){
      image_transpose[i][j] = normalized_image[j][i];
    }
  }
  for( m = 0; m < size_of_trainingset; m++ ){
    for( i = 0; i < size_of_trainingset; i++ ){
      covariance_image_matrix[i][m] = 0;
      for( j = 0; j < size_of_vector; j++ ){
        covariance_image_matrix[i][m] += normalized_image[m][j] * image_transpose[j][i];
      }
    }
  }
  for( i = 0; i < size_of_vector; i++ ){
    delete[] image_transpose[i];
  }
  delete[] image_transpose;
}


int decompose_eigen_image( float **& covariance_matrix,
                         int total_training_images,
                         float *& values_eigen,
                         float **& vector_eigen ){
  float threshold, theta, toug, mat_t, smat, mat_s, mat_h, mat_g, mat_c, mat_p, *mat_b, *mat_z;
  int jacobi_loop;
  int max_jacobi_iterations = 600;//max iterations for Jacobi decomposition algorithm
  mat_b = new float[total_training_images * sizeof(float)];
  mat_z = new float[total_training_images * sizeof(float)];
  
  //eigen vector and eigen value initialization
  for( int p = 0; p < total_training_images; p++ ){
    for ( int q = 0; q < total_training_images; q++){
      vector_eigen[p][q] = 0.0;
    }
    vector_eigen[p][p] = 1.0;
  }
  for( int i = 0; i < total_training_images; i++ ){
    mat_b[i] = values_eigen[i] = covariance_matrix[i][i];
    mat_z[i] = 0.0;
  }
  jacobi_loop = 0;
  for( int m = 0; m < max_jacobi_iterations; m++ ){
    smat = 0.0;
    for( int p = 0; p < total_training_images; p++ ){
      for( int q = p + 1; q < total_training_images; q++ ){
        smat += fabs(covariance_matrix[p][q]);
      }
    }
    if( smat == 0.0 ){
      
      //Sort the eigen values
      
      for( int i = 0; i < total_training_images; i++ ){
        int temp;
        mat_p = values_eigen[temp = i];
        for( int j = i + 1; j < total_training_images; j++ ){
          if( values_eigen[j] >= mat_p ){
            mat_p = values_eigen[temp = j];
          }
        }
        if( temp != i ){
          values_eigen[temp] = values_eigen[i];
          values_eigen[i] = mat_p;
          for( int j = 0; j < total_training_images; j++ ){
            mat_p = vector_eigen[j][i];
            vector_eigen[j][i] = vector_eigen[j][temp];
            vector_eigen[j][temp] = mat_p;
          }
        }
      }
      //Restoration of symmetric matrix
      for( int i = 1; i < total_training_images; i++ ){
        for( int j = 0; j < i; j++ ){
          covariance_matrix[j][i] = covariance_matrix[i][j];
        }
      }
      delete mat_z;
      delete mat_b;
      return jacobi_loop;
    }
    threshold = ( m < 4 ? 0.2 * smat / (total_training_images * total_training_images) : 0.0 );
    for( int p = 0; p < total_training_images; p++ ){
      for( int q = p + 1; q < total_training_images; q++ ){
        mat_g = 100.0 * fabs(covariance_matrix[p][q]);
        if( m > 4 &&
            fabs(values_eigen[p]) + mat_g == fabs(values_eigen[p]) &&
            fabs(values_eigen[q]) + mat_g == fabs(values_eigen[q]) ){
          covariance_matrix[p][q] = 0.0;
        } else if( fabs(covariance_matrix[p][q]) > threshold ){
          mat_h = values_eigen[q] - values_eigen[p];
          if( fabs(mat_h) + mat_g == fabs(mat_h) ){
            mat_t = covariance_matrix[p][q] / mat_h;
          } else {
            theta = 0.5 * mat_h / covariance_matrix[p][q];
            mat_t = 1.0 / ( fabs(theta) + sqrt( 1.0 + theta * theta ) );
            if( theta < 0.0 ){
              mat_t = -mat_t;
            }
          }
          mat_c = 1.0 / sqrt(1 + mat_t * mat_t);
          mat_s = mat_t * mat_c;
          toug = mat_s / (1.0 + mat_c);
          mat_h = mat_t * covariance_matrix[p][q];
          mat_z[p] -= mat_h;
          mat_z[q] += mat_h;
          values_eigen[p] -= mat_h;
          values_eigen[q] += mat_h;
          covariance_matrix[p][q] = 0.0;
        #define ROTATE_MATRIX(M,i,j,k,l) mat_g = M[i][j]; mat_h = M[k][l];
        for( int j = 0; j < p; j++ ){
          ROTATE_MATRIX( covariance_matrix, j, p, j, q );
        }
        for( int j = p + 1; j < q; j++ ){
          ROTATE_MATRIX( covariance_matrix, p, j, j, q );
        }
        for( int j = q + 1; j < total_training_images; j++ ){
          ROTATE_MATRIX( covariance_matrix, p, j, q, j );
        }
        for( int j = 0; j < total_training_images; j++ ){
          ROTATE_MATRIX( vector_eigen, j, p, j, q );
        }
        jacobi_loop++;
        }
      }
    }
    for( int p = 0; p < total_training_images; p++ ){
      mat_b[p] += mat_z[p];
      values_eigen[p] = mat_b[p];
      mat_z[p] = 0.0;
    }
  }
  delete mat_z;
  delete mat_b;
  return -1;
}


float ** project_eigen_vector( float **& normalized_matrix,
                        float **& vectors_eigen,
                        int size_of_vector,
                        int size_of_trainingset){
  float value = 0, mag,
    **vector_projection,
    **matrix_transpose;
    
    vector_projection = new float * [size_of_trainingset];
    for( int i = 0; i < size_of_trainingset; i++ ){
      vector_projection[i] = new float[size_of_vector];
    }
    matrix_transpose = new float * [size_of_vector];
    for( int i = 0; i < size_of_vector; i++ ){
      matrix_transpose[i] = new float[size_of_trainingset];
    }
    for( int i = 0; i < size_of_vector; i++ ){
      for( int j = 0; j < size_of_trainingset; j++ ){
        matrix_transpose[i][j] = normalized_matrix[j][i];
      }
    }
    FILE *fp=fopen( "eigen.vectors.dat", "w+b" );
    for( int k = 0; k < size_of_trainingset; k++ ){
      for( int i = 0; i < size_of_vector; i++ ){
        for( int j = 0; j < size_of_trainingset; j++ ){
          value += matrix_transpose[i][j] * vectors_eigen[j][k];
        }
        vector_projection[k][i] = value;
        value = 0;
      }
      /* eigenfood normalization */
      mag = 0;
      for( int l = 0; l < size_of_vector; l++ ){
        mag += vector_projection[k][l] * vector_projection[k][l];
      }
      mag = sqrt(mag);
      for( int l = 0; l < size_of_vector; l++ ){
        if( mag > 0) vector_projection[k][l] /= mag;
      }
      /* save the projected eigenvector */
      for( int p = 0; p < size_of_vector; p++ ){
        fwrite( &vector_projection[k][p], sizeof(float), 1, fp );
      }
    }
    fclose(fp);
    return vector_projection;
}

void build_eigen_databse_test(const string &image_file){
  eigen_database_table db;
  //int k=mean.width() * mean.height();
  int k=80;
  CImg<float> input_image(image_file.c_str());
  //CImg<float> *gray = new CImg<float> (input_image.get_RGBtoHSI().get_channel(2));
  CImg<float> *gray=grey_scale(input_image);
  //CImg<float> gray = input_image.get_RGBtoHSI().get_channel(2);
  gray->resize(40,40);
  CImg<float> mean( 40, 40,1,1);
  cout<<image_file.c_str()<<endl;
  float **norm_matrix,**covar_matrix,**eigenvectors,*eigenvalues,**eigenprojections;
  string path = image_file;
  std::size_t found = path.find_last_of("/\\");
  std::string str=path.substr(found+1,path.length()-1);
  img_information img;
  img.name = atoi(str.c_str());
  mean_normalized_image(*gray, mean, 1 );
  norm_matrix = new float * [1];
  norm_matrix[0] = new float[ mean.width() * mean.height()];
  image_normalization(*gray, 0, mean, norm_matrix );
  fflush(stdout);
  covar_matrix = new float * [ 1 ];
  covar_matrix[0] = new float[ 1];
  create_covariance_matrix( norm_matrix, covar_matrix, mean.width() * mean.height(),
                    1 );
  eigenvalues = new float [1];
  eigenvectors = new float * [1];
  eigenvectors[0] = new float [1];
  decompose_eigen_image( covar_matrix, 1, eigenvalues,
                                 eigenvectors );
  eigenprojections = project_eigen_vector( norm_matrix, eigenvectors,k,1);
  FILE *fp = fopen( "eigen.values.dat", "w+b" );
  for( int i = 0; i < 1; i++ ){
    fwrite( &eigenvalues[i], sizeof(float), 1, fp );
  }
  fclose(fp);
  db.food = 1;
  db.width = mean.width();
  db.height = mean.height();
  fp = fopen( "eigen.db.dat", "w+b" );
  fwrite( &db, sizeof(eigen_database_table), 1, fp );
  fclose(fp);
  fp = fopen( "eigen.img_information.dat", "w+b" );
  for( int i = 0; i < db.food; i++ ){
    img_information img_info = img;
    fwrite( &img_info, sizeof(img_information), 1, fp );
  }
  fclose(fp);
}

void build_eigen_databse( vector<string> path_of_images ){
  eigen_database_table db;
  //int k=mean.width() * mean.height();
  int k=80;
  vector< CImg<float> * > trainingset;
  vector< img_information > trainingset_info;
  /* expects 80 x 80 image. TODO: generalize */
  CImg<float> mean( 40, 40,1,1); 
  float ** norm_matrix,
  ** covar_matrix, 
  **eigenvectors,
  *eigenvalues,
  **eigenprojections;
  int i, j;
  for(int j=0;j<path_of_images.size();j++)
  {
    string path = path_of_images[j];
    std::size_t found = path.find_last_of("/\\");
    std::string str=path.substr(found+1,path.length()-1);
    img_information img;
    img.name = atoi(str.c_str());
    fflush(stdout);
    CImg<float> input_image(path.c_str());
    //CImg<float> *gray = new CImg<float> (input_image.get_RGBtoHSI().get_channel(2));
    CImg<float> *gray=grey_scale(input_image);
    gray->resize(40,40);
    trainingset.push_back( gray);
    trainingset_info.push_back( img );
  }
  for( int k = 0; k < trainingset_info.size(); k++ ){
    trainingset_info[k].index = k;
  }
  for( i = 0; i < trainingset.size(); i++ ){
    mean_normalized_image( *trainingset[i], mean, trainingset.size() );
  }
  mean.save("eigen.mean.bmp");
   norm_matrix = new float * [ trainingset.size() ];
  for( i = 0; i < trainingset.size(); i++ ){
     norm_matrix[i] = new float[ mean.width() * mean.height() ];
  }
  for( i = 0; i < trainingset.size(); i++ ){
    image_normalization( *trainingset[i], i, mean,  norm_matrix );
  }
   covar_matrix = new float * [ trainingset.size() ];
  for( i = 0; i < trainingset.size(); i++ ){
     covar_matrix[i] = new float[ trainingset.size() ];
  }
  create_covariance_matrix(  norm_matrix,  covar_matrix, mean.width() * mean.height(),
                    trainingset.size() );
  eigenvalues = new float [trainingset.size()];
  eigenvectors = new float * [trainingset.size()];
  for( i = 0; i < trainingset.size(); i++ ){
    eigenvectors[i] = new float [trainingset.size()];
  }
  decompose_eigen_image(  covar_matrix, trainingset.size(), eigenvalues, eigenvectors );
  eigenprojections = project_eigen_vector(  norm_matrix, eigenvectors,k, trainingset.size());
  FILE *fp = fopen( "eigen.values.dat", "w+b" );
  for( i = 0; i < trainingset.size(); i++ ){
    fwrite( &eigenvalues[i], sizeof(float), 1, fp );
  }
  fclose(fp);
  db.food = trainingset.size();
  db.width = mean.width();
  db.height = mean.height();
  fp = fopen( "eigen.db.dat", "w+b" );
  fwrite( &db, sizeof(eigen_database_table), 1, fp );
  fclose(fp);
  fp = fopen( "eigen.img_information.dat", "w+b" );
  for( int i = 0; i < db.food; i++ ){
    img_information img_info = trainingset_info.at(i);
    fwrite( &img_info, sizeof(img_information), 1, fp );
  }
  fclose(fp);
}

void load_eigen_database( eigen_database_table *db ){
  FILE *fp = fopen( "eigen.db.dat", "rb" );
  fread( db, sizeof(eigen_database_table), 1, fp );
  fclose(fp);
}
void load_eigen_information( vector<img_information> *info, eigen_database_table& db ){
  FILE *fp = fopen( "eigen.img_information.dat", "rb" );
  for( int i = 0; i < db.food; i++ ) {
    img_information img_info;
    fread( &img_info, sizeof(img_information), 1, fp );
    info->push_back(img_info);
  }
  fclose(fp);
}

float **load_eigen_vectors( eigen_database_table& db , int k){
  float **projections;
  int i, j;
  projections = new float * [db.food];
  for( i = 0; i < db.food; i++ ){
    projections[i] = new float[db.width * db.height];
  }
  FILE *fp = fopen( "eigen.vectors.dat", "rb" );
  for( i = 0; i < db.food; i++ ){
    for( j = 0; j < k; j++ ){
      fread( &projections[i][j], sizeof(float), 1, fp );
    }
  }
  fclose(fp);
  return projections;
}

CImg <float> *grey_scale(CImg <float> &image)
{
  
  for(int i=0;i<image.width();i++)
  {
    for(int j=0;j<image.height();j++)
    {
      const float valR=image(i,j,0,0);
      const float valG=image(i,j,0,0);
      const float valB=image(i,j,0,0);
      const float avg=(valR + valG + valB)/3;
      image(i,j,0)=image(i,j,1)=image(i,j,2)=avg;
    }
  }
  return &image;
}


