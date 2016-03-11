//
// Created by Tousif Ahmed on 3/10/16.
//

#ifndef CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H
#define CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H

#endif //CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H


class SVM : public Classifier
{
public:
    SVM(const vector<string> &_class_list) : Classifier(_class_list) {}

    // Nearest neighbor training. All this does is read in all the images, resize
    // them to a common size, convert to greyscale, and dump them as vectors to a file
    virtual void train(const Dataset &filenames)
    {
        ofstream datafile;
        datafile.open("train.dat");

        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
            cout << "SVM: Processing " << c_iter->first << endl;
           // CImg<double> class_vectors(size*size, filenames.size(), 1);


            // convert each image to be a row of this "model" image
            int target=distance(filenames.begin(), c_iter)+1;

            for(int i=0; i<c_iter->second.size(); i++)
            {
                vector<double> features=extract_features(c_iter->second[i].c_str());
                datafile<<target<<" ";
                for(int j=0;j<features.size();j++)
                {
                    datafile<<(j+1)<<":"<<features.at(j)<<" ";
                }
                datafile<<"\n";

            }

        }
        datafile.close();
      //  system("cd svm_multiclass/");
        chdir("svm_multiclass/");
        system("make");
        system("./svm_multiclass_learn -c 100 ../train.dat ../svm_model >../train.txt");
    }

    virtual string classify(const string &filename)
    {

        cout<<"In Classification"<<endl;
        ofstream datafile;
        datafile.open("test.dat");
        vector<double> features = extract_features(filename);
        datafile<<"1 ";
        for(int j=0;j<features.size();j++)
        {
            datafile<<(j+1)<<":"<<features.at(j)<<" ";
        }
        datafile<<"\n";
        datafile.close();

        chdir("svm_multiclass/");
        //   system("cd svm_multiclass");
        system("make > garbage.txt");
        //  system("ls -l >pwd.txt ");
        system("./svm_multiclass_classify ../test.dat ../svm_model ../test_output.txt >../test.txt ");
        chdir ("../");

        string str;
        string output_label;
        ifstream outputfile("test_output.txt");
        while (std::getline(outputfile, str))
        {
            int index=str.find(" ");
            int c_index=stoi(str.substr(0,index));
           // cout<<class_list[c_index-1]<<endl;
            output_label=class_list[c_index-1];
            //cout<<str<<endl;
        }

        return output_label;

    }

    virtual void load_model()
    {
        return;

    }
 /*   virtual void test(const Dataset &filenames)
    {
        cout<<"In SVM test "<<endl;
        ofstream datafile;
        datafile.open("test.dat");

        for(Dataset::const_iterator c_iter=filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
            cout << "SVM: Processing " << c_iter->first << endl;
            int target=distance(filenames.begin(), c_iter)+1;

            for(int i=0; i<c_iter->second.size(); i++)
            {
                vector<double> features=extract_features(c_iter->second[i].c_str());
                datafile<<target<<" ";
                for(int j=0;j<features.size();j++)
                {
                    datafile<<(j+1)<<":"<<features.at(j)<<" ";
                }
                datafile<<"\n";
            }

        }
        datafile.close();
        datafile.close();
        chdir("svm_multiclass/");
     //   system("cd svm_multiclass");
        system("make");
      //  system("ls -l >pwd.txt ");
        system("./svm_multiclass_classify ../test.dat ../svm_model ../output_file.txt >../test.txt ");

        chdir("../");
        cout << ifstream("test.txt").rdbuf();

        string str;
        ifstream outputfile("output_file.txt");
        while (std::getline(outputfile, str))
        {
            int index=str.find(" ");
            int c_index=stoi(str.substr(0,index));
            cout<<class_list[c_index-1]<<endl;
        }


    }*/



protected:
    // extract features from an image, which in this case just involves resampling and
    // rearranging into a vector of pixel data.
    vector<double> extract_features(const string &filename)
    {
        // For Color, Make it 3

        CImg<double> image(filename.c_str());
        CImg<double> gray_image(image.width(),image.height(),1,1);
        vector<double>  feature_vector;
        if(image.spectrum() == 3 && grayscale==1)
        {
            gray_image= image.get_RGBtoHSI().get_channel(2);
            image=gray_image;
        }

        image=gray_image;
        image.normalize(0,255).save("output_gray.png");
        image.resize(size,size,1,grayscale);
        for(int c=0;c<grayscale;c++)
        {
            for(int i=0;i<image.width();i++)
            {
                for(int j=0;j<image.height();j++)
                {
                    feature_vector.push_back(normalize(image(i,j,0,c)));
                }
            }
        }

        return feature_vector;
      //  return (CImg<double>(filename.c_str())).resize(size,size,1,3).unroll('x');
    }
    double normalize(double value)
    {
        return value/255.0;

    }
    static const int size=20;  // subsampled image resolution
    static const int grayscale=1;  // Test with Grayscale and Color
    map<string, CImg<double> > models; // trained models
    vector<double> classes;
};