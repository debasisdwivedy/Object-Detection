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
void MultiplyingMatrix(const vector<vector<double> >&, const vector<vector<double> >&, vector< vector<double> > &);
void MatrixTranspose(vector<vector<double> >&, vector<vector<double> >&);
typedef struct {
  int food;
  int width;
  int height;
} eigen_db_t;


typedef struct {
  int index;
  int subject;
} image_info;


void process_image( CImg<float>& image,
                    CImg<float>& mean,
                    int trainingset_size ){
  int i, j;
  //CImg<float> gray = image.get_RGBtoHSI().get_channel(2);
  //cout<<"Before resize"<<endl;
  image.resize(40,40,1,1);
  //cout<<"After resize"<<endl;
  for( i = 0; i < image.width(); i++ ){
    for( j = 0; j < image.height(); j++ ){
      *mean.data( i, j ) += *image.data( i, j ) / (trainingset_size);
    }
  }
  cout<<"end of process image"<<endl;
}

void normalize_image( CImg<float>& image,
                      int index,
                      CImg<float>& mean,
                      float **&normalized_m ){
  int i, j, k;
  for( i = 0, k = 0; i < image.width(); i++ ){
    for( j = 0; j < image.height(); j++, k++ ){
      normalized_m[index][k] = *image.data( i, j ) - *mean.data( i, j );
    }
  }
  cout<<"end of normalize_image"<<endl;
}

void eigen_covariance( float **& normalized_m,
                       float **& covariance_m,
                       int vector_size,
                       int trainingset_size ){
  float **transpose;
  int i, j, m;
  transpose = new float * [vector_size];
  for( i = 0; i < vector_size; i++ ){
    transpose[i] = new float [trainingset_size];
  }
  for( i = 0; i < vector_size; i++ ){
    for( j = 0; j < trainingset_size; j++ ){
      transpose[i][j] = normalized_m[j][i];
    }
  }
  for( m = 0; m < trainingset_size; m++ ){
    if( m % 30 == 0 ){
      printf( "step 4/6 (covariance): image %d of %d\n", m, trainingset_size );
      fflush(stdout);
    }
    for( i = 0; i < trainingset_size; i++ ){
      covariance_m[i][m] = 0;
      for( j = 0; j < vector_size; j++ ){
        covariance_m[i][m] += normalized_m[m][j] * transpose[j][i];
      }
    }
  }
  for( i = 0; i < vector_size; i++ ){
    delete[] transpose[i];
  }
  delete[] transpose;
}


int eigen_decomposition( float **& matrix,
                         int m_size,
                         float *& eigenvalues,
                         float **& eigenvectors ){
  float threshold, theta, tau, t, sm, s, h, g, c, p, *b, *z;
  int jiterations;
  /* max iterations in Jacobi decomposition algorithm */
  int jacobi_max_iterations = 500;
  b = new float[m_size * sizeof(float)];
  z = new float[m_size * sizeof(float)];
  /* initialize eigenvectors and eigen values */
  for( int ip = 0; ip < m_size; ip++ ){
    for ( int iq = 0; iq < m_size; iq++){
      eigenvectors[ip][iq] = 0.0;
    }
    eigenvectors[ip][ip] = 1.0;
  }
  for( int ip = 0; ip < m_size; ip++ ){
    b[ip] = eigenvalues[ip] = matrix[ip][ip];
    z[ip] = 0.0;
  }
  jiterations = 0;
  for( int m = 0; m < jacobi_max_iterations; m++ ){
    printf("jacobi iteration %d\n", m);
    fflush(stdout);
    sm = 0.0;
    for( int ip = 0; ip < m_size; ip++ ){
      for( int iq = ip + 1; iq < m_size; iq++ ){
        sm += fabs(matrix[ip][iq]);
      }
    }
    if( sm == 0.0 ){
      /* eigenvalues & eigenvectors sorting */
      for( int i = 0; i < m_size; i++ ){
        int k;
        p = eigenvalues[k = i];
        for( int j = i + 1; j < m_size; j++ ){
          if( eigenvalues[j] >= p ){
            p = eigenvalues[k = j];
          }
        }
        if( k != i ){
          eigenvalues[k] = eigenvalues[i];
          eigenvalues[i] = p;
          for( int j = 0; j < m_size; j++ ){
            p = eigenvectors[j][i];
            eigenvectors[j][i] = eigenvectors[j][k];
            eigenvectors[j][k] = p;
          }
        }
      }
      /* restore symmetric matrix's matrix */
      for( int i = 1; i < m_size; i++ ){
        for( int j = 0; j < i; j++ ){
          matrix[j][i] = matrix[i][j];
        }
      }
      delete z;
      delete b;
      return jiterations;
    }
    threshold = ( m < 4 ? 0.2 * sm / (m_size * m_size) : 0.0 );
    for( int ip = 0; ip < m_size; ip++ ){
      for( int iq = ip + 1; iq < m_size; iq++ ){
        g = 100.0 * fabs(matrix[ip][iq]);
        if( m > 4 &&
            fabs(eigenvalues[ip]) + g == fabs(eigenvalues[ip]) &&
            fabs(eigenvalues[iq]) + g == fabs(eigenvalues[iq]) ){
          matrix[ip][iq] = 0.0;
        } else if( fabs(matrix[ip][iq]) > threshold ){
          h = eigenvalues[iq] - eigenvalues[ip];
          if( fabs(h) + g == fabs(h) ){
            t = matrix[ip][iq] / h;
          } else {
            theta = 0.5 * h / matrix[ip][iq];
            t = 1.0 / ( fabs(theta) + sqrt( 1.0 + theta * theta ) );
            if( theta < 0.0 ){
              t = -t;
            }
          }
          c = 1.0 / sqrt(1 + t * t);
          s = t * c;
          tau = s / (1.0 + c);
          h = t * matrix[ip][iq];
          z[ip] -= h;
          z[iq] += h;
          eigenvalues[ip] -= h;
          eigenvalues[iq] += h;
          matrix[ip][iq] = 0.0;
#define M_ROTATE(M,i,j,k,l) g = M[i][j]; h = M[k][l];
          //M[i][j] = g - s * (h + g * tau); M[k][l] = h + s * (g - h * tau)
        for( int j = 0; j < ip; j++ ){
          M_ROTATE( matrix, j, ip, j, iq );
        }
        for( int j = ip + 1; j < iq; j++ ){
          M_ROTATE( matrix, ip, j, j, iq );
        }
        for( int j = iq + 1; j < m_size; j++ ){
          M_ROTATE( matrix, ip, j, iq, j );
        }
        for( int j = 0; j < m_size; j++ ){
          M_ROTATE( eigenvectors, j, ip, j, iq );
        }
        jiterations++;
        }
      }
    }
    for( int ip = 0; ip < m_size; ip++ ){
      b[ip] += z[ip];
      eigenvalues[ip] = b[ip];
      z[ip] = 0.0;
    }
  }
  delete z;
  delete b;
  return -1;
}


float ** eigen_project( float **& m_normalized,
                        float **& eigenvectors,
                        float *& eigenvalues,
                        int vector_size,
                        int trainingset_size ,string mode){
  float value = 0, mag,
    **projections,
    **transpose;
    
    projections = new float * [trainingset_size];
    for( int i = 0; i < trainingset_size; i++ ){
      projections[i] = new float[vector_size];
    }
    transpose = new float * [vector_size];
    for( int i = 0; i < vector_size; i++ ){
      transpose[i] = new float[trainingset_size];
    }
    for( int i = 0; i < vector_size; i++ ){
      for( int j = 0; j < trainingset_size; j++ ){
        transpose[i][j] = m_normalized[j][i];
      }
    }
    /*if(mode.compare("train") == 0)
    {
    *fp = fopen( "eigen.vectors.train.dat", "w+b" );
    }
    else
    {
      *fp = fopen( "eigen.vectors.test.dat", "w+b" );
    }*/
    FILE *fp=fopen( "eigen.vectors.dat", "w+b" );
    for( int k = 0; k < trainingset_size; k++ ){
      if( k % 30 == 0 ){
        printf( "step 5/6 (projection): image %d of %d\n", k, trainingset_size );
        fflush(stdout);
      }
      for( int i = 0; i < vector_size; i++ ){
        for( int j = 0; j < trainingset_size; j++ ){
          value += transpose[i][j] * eigenvectors[j][k];
        }
        projections[k][i] = value;
        value = 0;
      }
      /* eigenfood normalization */
      mag = 0;
      for( int l = 0; l < vector_size; l++ ){
        mag += projections[k][l] * projections[k][l];
      }
      mag = sqrt(mag);
      for( int l = 0; l < vector_size; l++ ){
        if( mag > 0) projections[k][l] /= mag;
      }
      /* save the projected eigenvector */
      for( int p = 0; p < vector_size; p++ ){
        fwrite( &projections[k][p], sizeof(float), 1, fp );
      }
    }
    fclose(fp);
    return projections;
}

float *eigen_weights( CImg<float>& face,
                      CImg<float>& mean,
                      float **& projections,
                      int trainingset_size ){
  CImg<float> normalized = face - mean;
  int m, n, i, index;
  float *weights, w;
  weights = new float[trainingset_size];
  for( index = 0; index < trainingset_size; index++ ){
    w = 0.0;
    for( m = 0, i = 0; m < normalized.width(); m++ ){
      for( n = 0; n < normalized.height(); n++, i++ ){
        w += projections[index][i] * *normalized.data(m,n);
      }
    }
    weights[index] = w;
  }
  return weights;
}

void eigen_build_db_test(const string &image_file){
  CImg<float> input_image(image_file.c_str());
  CImg<float> mean( 40, 40, 1, 1 );
  cout<<image_file.c_str()<<endl;
  float **normalized_m,**covariance_m,**eigenvectors,*eigenvalues,**eigenprojections;
  string path = image_file;
  std::size_t found = path.find_last_of("/\\");
  std::string str=path.substr(found+1,path.length()-1);
  image_info img;
  img.subject = atoi(str.c_str());
  fflush(stdout);
  process_image(input_image, mean, 1 );
  normalized_m = new float * [1];
  normalized_m[0] = new float[ mean.width() * mean.height()];
  normalize_image(input_image, 0, mean, normalized_m );
  fflush(stdout);
  covariance_m = new float * [ 1 ];
  covariance_m[0] = new float[ 1];
  eigen_covariance( normalized_m, covariance_m, mean.width() * mean.height(),
                    1 );
  eigenvalues = new float [1];
  eigenvectors = new float * [1];
  eigenvectors[0] = new float [1];
  int ret = eigen_decomposition( covariance_m, 1, eigenvalues,
                                 eigenvectors );
  eigenprojections = eigen_project( normalized_m, eigenvectors, eigenvalues,
                                    mean.width() * mean.height(),
                                    1 ,"test");
  fflush(stdout);
}

void eigen_build_db( vector<string> path_of_images ){
  eigen_db_t db;
  vector< CImg<float> * > trainingset;
  vector< image_info > trainingset_info;
  /* expects 40 x 40 image. TODO: generalize */
  CImg<float> mean( 40, 40, 1, 1 ); // mean face from training set
  float **normalized_m, // normalized food matrix A
  **covariance_m, // covariance matrix (C = tA*A --> C = A*tA)
  **eigenvectors, // eigenvectors of jacobi decomposition
  *eigenvalues, // eigenvalues of jacobi decomposition
  **eigenprojections, // eigenvectors projected to face space
  **eigenweights; // weights
  int i, j;
  printf( "--- Building Eigen food database ---\n" );
  printf( "\t@ Loading training set ...\n" );
  for(int j=0;j<path_of_images.size();j++)
  {
    string path = path_of_images[j];
    std::size_t found = path.find_last_of("/\\");
    std::string str=path.substr(found+1,path.length()-1);
    image_info img;
    img.subject = atoi(str.c_str());
    fflush(stdout);
    trainingset.push_back( new CImg<float>( path.c_str() ) );
    trainingset_info.push_back( img );
  }
  std::vector< std::vector< double > > eigen_vector_matrix (trainingset.size(),std::vector<double> (trainingset.size(),0));
  std::vector< std::vector< double > > eigen_vector_transpose_matrix(trainingset.size(),std::vector<double> (trainingset.size(),0));
  std::vector< std::vector< double > > eigen_vector_transpose_multiply_matrix(trainingset.size(),std::vector<double> (trainingset.size(),0));
  for( int k = 0; k < trainingset_info.size(); k++ ){
    trainingset_info[k].index = k;
  }
  printf( "\t@ Processing mean face ...\n" );
  for( i = 0; i < trainingset.size(); i++ ){
    printf( "step 2/6 (processing): image %d; subject %d", i,
            trainingset_info[i].subject);
    fflush(stdout);
    process_image( *trainingset[i], mean, trainingset.size() );
  }
  mean.save("eigen.mean.bmp");
  printf( "\t@ Normalizing food ...\n" );
  normalized_m = new float * [ trainingset.size() ];
  for( i = 0; i < trainingset.size(); i++ ){
    normalized_m[i] = new float[ mean.width() * mean.height() ];
  }
  for( i = 0; i < trainingset.size(); i++ ){
    normalize_image( *trainingset[i], i, mean, normalized_m );
    printf( "step 3/6 (normalizing): image %d; subject %d",
            i, trainingset_info[i].subject);
    fflush(stdout);
  }
  printf( "\t@ Computing covariance ...\n" );
  covariance_m = new float * [ trainingset.size() ];
  for( i = 0; i < trainingset.size(); i++ ){
    covariance_m[i] = new float[ trainingset.size() ];
  }
  cout<<"Training set size:-"<<trainingset.size()<<endl;
  eigen_covariance( normalized_m, covariance_m, mean.width() * mean.height(),
                    trainingset.size() );
  printf( "\t@ Computing Jacobi decomposition ... " );
  fflush(stdout);
  eigenvalues = new float [trainingset.size()];
  eigenvectors = new float * [trainingset.size()];
  for( i = 0; i < trainingset.size(); i++ ){
    eigenvectors[i] = new float [trainingset.size()];
  }
  int ret = eigen_decomposition( covariance_m, trainingset.size(), eigenvalues,
                                 eigenvectors );
  /*for( int i = 0; i < trainingset.size(); i++ ){
    for( int j = 0; j < trainingset.size(); j++ ){
      cout<<eigenvectors[i][j];
      eigen_vector_matrix[i][j]=eigenvectors[i][j];
    }
    cout<<endl;
  }*/
  
  printf( "[%d]\n", ret );
  MatrixTranspose(eigen_vector_matrix,eigen_vector_transpose_matrix);
  /*for( int i = 0; i < trainingset.size(); i++ ){
    for( int j = 0; j < trainingset.size(); j++ ){
      cout<<eigen_vector_transpose_matrix[i][j];
    }
    cout<<endl;
  }*/
  MultiplyingMatrix(eigen_vector_matrix,eigen_vector_transpose_matrix,eigen_vector_transpose_multiply_matrix);
  /*for( int i = 0; i < trainingset.size(); i++ ){
    for( int j = 0; j < trainingset.size(); j++ ){
      cout<<eigen_vector_transpose_multiply_matrix[i][j];
    }
    cout<<endl;
  }*/
  printf( "\t@ Projecting eigenvectors ...\n" );
  eigenprojections = eigen_project( normalized_m, eigenvectors, eigenvalues,
                                    mean.width() * mean.height(),
                                    trainingset.size() ,"train");
  printf( "\t@ Saving eigen values ...\n" );
  fflush(stdout);
  FILE *fp = fopen( "eigen.values.dat", "w+b" );
  for( i = 0; i < trainingset.size(); i++ ){
    fwrite( &eigenvalues[i], sizeof(float), 1, fp );
  }
  fclose(fp);
  printf( "\t@ Computing eigen weights ...\n" );
  fflush(stdout);
  eigenweights = new float * [trainingset.size()];
  for( i = 0; i < trainingset.size(); i++ ){
    if( i % 30 == 0 ){
      printf( "step 6/6 (eigen weights): image %d of %d; subject %d",
              i, trainingset.size(), trainingset_info[i].subject);
      fflush(stdout);
    }
    eigenweights[i] = eigen_weights( *trainingset[i], mean, eigenprojections,
                                     trainingset.size() );
  }
  
  /*for( i = 0; i < trainingset.size(); i++ ){
    for( j = 0; j < trainingset.size(); j++ ){
      cout<<eigenweights[i][j]<<endl;
    }
  }*/
  
  fp = fopen( "eigen.weights.dat", "w+b" );
  for( i = 0; i < trainingset.size(); i++ ){
    for( j = 0; j < trainingset.size(); j++ ){
      fwrite( &eigenweights[i][j], sizeof(float), 1, fp );
    }
  }
  fclose(fp);
  printf( "\t@ Saving db info ...\n" );
  fflush(stdout);
  db.food = trainingset.size();
  db.width = mean.width();
  db.height = mean.height();
  fp = fopen( "eigen.db.dat", "w+b" );
  fwrite( &db, sizeof(eigen_db_t), 1, fp );
  fclose(fp);
  fp = fopen( "eigen.image_info.dat", "w+b" );
  for( int i = 0; i < db.food; i++ ){
    image_info ii = trainingset_info.at(i);
    fwrite( &ii, sizeof(image_info), 1, fp );
  }
  cout<<"done"<<endl;
  fclose(fp);
}

void eigen_load_db( eigen_db_t *db ){
  printf("eigen: loading database\n");
  FILE *fp = fopen( "eigen.db.dat", "rb" );
  fread( db, sizeof(eigen_db_t), 1, fp );
  fclose(fp);
}
void eigen_load_info( vector<image_info> *info, eigen_db_t& db ){
  printf("eigen: loading image info\n");
  FILE *fp = fopen( "eigen.image_info.dat", "rb" );
  for( int i = 0; i < db.food; i++ ) {
    image_info ii;
    fread( &ii, sizeof(image_info), 1, fp );
    info->push_back(ii);
  }
  fclose(fp);
}

float **eigen_load_vectors( eigen_db_t& db ){
  printf("eigen: loading vectors\n");
  float **projections;
  int i, j;
  projections = new float * [db.food];
  for( i = 0; i < db.food; i++ ){
    projections[i] = new float[db.width * db.height];
  }
  FILE *fp = fopen( "eigen.vectors.dat", "rb" );
  for( i = 0; i < db.food; i++ ){
    for( j = 0; j < db.width * db.height; j++ ){
      fread( &projections[i][j], sizeof(float), 1, fp );
    }
  }
  fclose(fp);
  return projections;
}
void eigen_load_weights( eigen_db_t& db, float *weights ){
  printf("eigen: loading weights - size = %d\n", db.food);
  fflush(stdout);
  int i, j;
  FILE *fp = fopen( "eigen.weights.dat", "rb" );
  for( i = 0; i < db.food; i++ ){
    for( j = 0; j < db.food; j++ ){
      fread( &weights[i*db.food + j], sizeof(float), 1, fp );
    }
  }
  printf("finished loading weights\n");
  fflush(stdout);
  fclose(fp);
}
void eigen_build_iweights( vector< CImg<float> > food, CImg<float> mean,
                           float **projections, int size, float *iweights ){
  printf("eigen: building iweights\n");
  for( int i = 0; i < food.size(); i++ ) {
    CImg<float> normalized = food[i] - mean;
    for( int j = 0; j < size; j++) {
      float w = 0.0;
      for( int m = 0, ii = 0; m < normalized.width(); m++ ){
        for( int n = 0; n < normalized.height(); n++, ii++ ){
          w += projections[j][ii] * *normalized.data(m,n);
        }
      }
      iweights[i*size + j] = w;
    }
  }
  printf(" >>> finished building iweights\n");
  fflush(stdout);
}
void MultiplyingMatrix(const vector<vector<double> >& A, const vector<vector<double> >& B, vector< vector<double> > &C)
{
  int Arows=A.size();
  vector<double> tempA=A[1];
  int Acols=tempA.size();
  int Brows=B.size();
  vector<double> tempB=B[1];
  int Bcols=tempB.size();
  for (int i = 0; i < Arows; i++)
  {
    for (int j = 0; j < Bcols; j++)
    {
      for (int rwcl = 0; rwcl < Acols; rwcl++)
      {
        C[i][j] += (A[i][rwcl] * B[rwcl][j]);
      }
    }
  }
}

void MatrixTranspose(vector<vector<double> >& A, vector<vector<double> >& B)
{
  int rows=A.size();
  int cols=A[1].size();
  for(int i=0;i<cols;i++)
  {
    for(int j=0;j<rows;j++)
    {
      B[i][j]=A[j][i];
    }
  }
}


