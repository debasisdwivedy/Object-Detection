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

    			eigen_db_t db;
    			float **projections;
    			vector<image_info> trainingset_info;
    			vector<image_info> testingset_info;
    			image_info img;
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
    	            eigen_build_db(names_of_training_images);
    	            /* Load learned system from disk and allocate memory for new structures. */
    	                eigen_load_db(&db);
    	                eigen_load_info(&trainingset_info, db);
    	                projections = eigen_load_vectors(db);
    	                cout<<c_iter->second.size()<<endl;
    	                cout<<names_of_training_images.size()<<endl;
    	                for (int j = 0; j < names_of_training_images.size(); j++)
    	                {
    	                datafile << target << " ";
    	                for(int v=0;v<db.height * db.width;v++)
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
    eigen_db_t db;
    float **projections;
    ofstream datafile;
    int target = 1;
    datafile.open("eigen_vector_test.dat");
    cout<<"eigen_build_db_test"<<endl;
    eigen_build_db_test(filename);
    cout<<"eigen_load_db"<<endl;
    eigen_load_db(&db);
    cout<<"eigen_load_vectors"<<endl;
    projections = eigen_load_vectors(db);
    cout<<"eigen_load_vectors_end"<<endl;
    for (int j = 0; j < 1; j++) {
      datafile << target << " ";
      for (int v = 0; v < db.height * db.width; v++) {
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
      int c_index = stoi(str.substr(0, index));
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


