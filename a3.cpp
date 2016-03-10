// B657 assignment 3 skeleton code, D. Crandall
//
// Compile with: "make"
//
// This skeleton code implements nearest-neighbor classification
// using just matching on raw pixel values, but on subsampled "tiny images" of
// e.g. 20x20 pixels.
//
// It defines an abstract Classifier class, so that all you have to do
// :) to write a new algorithm is to derive a new class from
// Classifier containing your algorithm-specific code
// (i.e. load_model(), train(), and classify() methods) -- see
// NearestNeighbor.h for a prototype.  So in theory, you really
// shouldn't have to modify the code below or the code in Classifier.h
// at all, besides adding an #include and updating the "if" statement
// that checks "algo" below.
//
// See assignment handout for command line and project specifications.
//
#include "CImg.h"
#include <ctime>
#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>
#include <vector>
#include <Sift.h>
#include <sys/types.h>
#include <fstream>
#include <dirent.h>
#include <map>
#include <numeric>

//Use the cimg namespace to access the functions easily
using namespace cimg_library;
using namespace std;

// Dataset data structure, set up so that e.g. dataset["bagel"][3] is
// filename of 4th bagel image in the dataset
typedef map<string, vector<string> > Dataset;
string directory_name="";

void call_svm(Dataset filenames);


#include <Classifier.h>
#include <NearestNeighbor.h>
#include <svm.h>

// Figure out a list of files in a given directory.
//
vector<string> files_in_directory(const string &directory, bool prepend_directory = false)
{
  vector<string> file_list;
  DIR *dir = opendir(directory.c_str());
  if(!dir)
    throw std::string("Can't find directory " + directory);
  
  struct dirent *dirent;
  while ((dirent = readdir(dir))) 
    if(dirent->d_name[0] != '.')
      file_list.push_back((prepend_directory?(directory+"/"):"")+dirent->d_name);

  closedir(dir);
  return file_list;
}

int main(int argc, char **argv)
{
  try {
    if(argc < 3)
      throw string("Insufficent number of arguments");

    string mode = argv[1];
    string algo = argv[2];

    // Scan through the "train" or "test" directory (depending on the
    //  mode) and builds a data structure of the image filenames for each class.
    Dataset filenames; 
    vector<string> class_list = files_in_directory(mode);
    for(vector<string>::const_iterator c = class_list.begin(); c != class_list.end(); ++c)
      filenames[*c] = files_in_directory(mode + "/" + *c, true);

    // set up the classifier based on the requested algo
    Classifier *classifier=0;
    if(algo == "nn")
    {
      classifier = new NearestNeighbor(class_list);


    }
    else if (algo=="svm")
    {
        classifier=new SVM(class_list);

    }
    else if(algo=="test")
    {
      CImg<double > test("test.jpg");
      vector<double> oned;
      cout<<test.spectrum()<<endl;
      test.resize(8,8,1,1);
      test.save("resized.png");
      for(int i=0;i<test.height();i++)
      {
        for(int j=0;j<test.width();j++)
        {
          cout<<test(i,j,0,0)<<" ";
          oned.push_back(test(i,j,0,0));

        }

        cout<<endl;
      }
      test.unroll('x');
      for(int i=0;i<test.width();i++)
      {
        for(int j=0;j<test.height();j++)
          cout<<test(j,i,0,0)<<" ";
        cout<<endl;
      }
      for (int i=0;i<oned.size();i++)
      {
        cout<<oned.at(i)<<" ";
      }
      cout<<endl;



    }
    else
      throw std::string("unknown classifier " + algo);

    // now train or test!
    if(mode == "train")
      classifier->train(filenames);
    else if(mode == "test")
      classifier->test(filenames);
    else
      throw std::string("unknown mode!");

  }
  catch(const string &err) {
    cerr << "Error: " << err << endl;
  }
}


void call_svm(Dataset filenames)
{
  for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter) {
    cout << "Processing " << c_iter->first << endl;
  }
  system("ls -l >test.txt"); // execute the UNIX command "ls -l >test.txt"
 // cout << ifstream("test.txt").rdbuf();
}








