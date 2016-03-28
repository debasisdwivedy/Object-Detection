//
// Created by Debasis Dwivedy on 3/24/16.
//

#include <vector>
#include <eigen.h>
#ifndef CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H
#define CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H

#endif //CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H

class EIGEN_FOOD : public Classifier
{
public:
    EIGEN_FOOD(const vector <string> &_class_list) : Classifier(_class_list) { }

    virtual void train(const Dataset &filenames)
        {

    			eigen_database_table db;
          //int k=db.width*db.height;
          int k=80;
    			float **projections;
    			vector<img_information> trainingset_info;
    			vector<img_information> testingset_info;
    			img_information img;
    			ofstream datafile;
    	        datafile.open("eigen_vector_train.dat");

    	        // For each folder of images
    	        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
    	        {
    	            cout << "EIGEN: Processing " << c_iter->first << endl;
    	            int target = distance(filenames.begin(), c_iter) + 1;

    	            std::vector<string> names_of_training_images(c_iter->second.size());

    	            // For each image in the folder
    	            for (int i = 0; i < c_iter->second.size(); i++)
    	            {
    	            	names_of_training_images[i]=c_iter->second[i].c_str();
    	            }
    	            build_eigen_databse(names_of_training_images);
    	            // Load learned system from disk and allocate memory for new structures.
    	                load_eigen_database(&db);
    	                load_eigen_information(&trainingset_info, db);
    	                projections = load_eigen_vectors(db,k);
    	                for (int j = 0; j < names_of_training_images.size(); j++)
    	                {
    	                datafile << target << " ";
    	                for(int v=0;v<k;v++)
    	                {
    	                	datafile << (v + 1) << ":" << projections[j][v] << " ";
    	                }
    	                datafile << "\n";
    	                }

    	        }
    	        datafile.close();
    	        chdir("svm_multiclass/");
    	        system("make > garbage.txt ");
    	        system("./svm_multiclass_learn -c 50 ../eigen_vector_train.dat ../eigen_model >../eigen_train.txt");
    	        chdir("../");

        }

  virtual string classify(const string &filename) {
    cout<<"start"<<endl;
    //int k=db.width*db.height;
    int k=80;
    eigen_database_table db;
    float **projections;
    ofstream datafile;
    int target = 1;
    img_information img;
    vector<img_information> testingset_info;
    datafile.open("eigen_vector_test.dat");
    build_eigen_databse_test(filename);
    
    load_eigen_database(&db);
    projections = load_eigen_vectors(db,k);
    for (int j = 0; j < 1; j++) {
      datafile << target << " ";
      for (int v = 0; v < k; v++) {
        datafile << (v + 1) << ":" << projections[j][v] << " ";
      }
      datafile << "\n";
    }
    datafile.close();
    cout<<"done classifing"<<endl;
    // Classify images using SVM_Multiclass
    chdir("svm_multiclass/");
    system("make > garbage.txt");
    system(
      "./svm_multiclass_classify ../eigen_vector_test.dat ../eigen_model ../eigen_test_output.txt >../eigen_test.txt ");
    chdir("../");
    
    string str;
    string output_label;
    ifstream outputfile("eigen_test_output.txt");
    while (std::getline(outputfile, str)) {
      int index = str.find(" ");
      string p=str.substr(0,index);
      int c_index=atoi(p.c_str());
      //int c_index = atoi(str.substr(0, index));
      // cout<<class_list[c_index-1]<<endl;
      output_label = class_list[c_index - 1];
      //cout<<str<<endl;
    }
    return output_label;
    
  }

    virtual void load_model()
        {
            return;

        }
};


