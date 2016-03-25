//
// Created by Tousif Ahmed on 3/10/16.
//

#ifndef CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H
#define CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H

#endif //CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H


class SVM : public Classifier
{
    int feature_type;
public:
    SVM(const vector<string> &_class_list,int feature) : Classifier(_class_list) {feature_type=feature;}

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
        system("make > garbage.txt ");

        system("./svm_multiclass_learn -c 50 t 2 ../train.dat ../svm_model >../train.txt");
    }

    virtual string classify(const string &filename)
    {

     //   cout<<"In Classification"<<endl;
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
	     string s=str.substr(0,index);
            int c_index=atoi(s.c_str());
            //int c_index=stoi(str.substr(0,index));
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
    vector<double> deep_features(const string &filename)
    {
        vector<double>  feature_vector;
        cout<< filename<<endl;
        string command="./overfeat/bin/linux_64/overfeat -f "+filename+" > deep_features.txt";
        cout<<command<<endl;
	system(command.c_str());
        
       return feature_vector; 
        
    }


    vector<double> haar_features(CImg<double> image)
    {
        //cout<< "Haar Features"<<endl;

        // CIMG initialize
        CImg<double> gray_image(image.width(),image.height(),1,1);
        CImg<double> integral_image(image.width(),image.height(),1,1);
        CImg<double> row_sum(image.width(),image.height(),1,1);


        // Features
       // Followed http://stackoverflow.com/questions/1707620/viola-jones-face-detection-claims-180k-features
        int features=5;
        int feature[5][2] = {{2,1}, {1,2}, {3,1}, {1,3}, {2,2}};
        int max_size=24;
        int no_of_features=1000;

        srand (time(NULL));


        vector<double>  feature_vector;
        int grayscale=1;  // Test with Grayscale and Color
        if(image.spectrum() == 3 && grayscale==1)
        {
            gray_image= image.get_RGBtoHSI().get_channel(2);
            image=gray_image;
        }
        image.normalize(0,255).save("test.jpg");

        for(int x=0;x<image.width();x++)
        {
            for(int y=0;y<image.height();y++)
            {

                row_sum(x,y,0,0)=y>0?row_sum(x,y-1,0,0)+image(x,y,0,0):image(x,y,0,0);
                integral_image(x,y,0,0)=x>0?row_sum(x,y,0,0)+integral_image(x-1,y,0,0):row_sum(x,y,0,0);
            }
        }

        // Feature Generation
        for(int i=0;i<no_of_features;i++)
        {
            int which_feature = rand() % 5;
            int size_x = feature[which_feature][0];
            int size_y = feature[which_feature][1];

            int max=size_x>size_y?size_x:size_y;

            int r_size_x=rand()%(max_size/size_x)+1;
            int r_size_y=rand()%(max_size/size_y)+1;

            size_x=size_x*r_size_x;
            size_y=size_y*r_size_y;
            int black=rand()%2; // 0 black, 1 white

            int pos_x=rand()%(image.width()-size_x-1)+size_x;
            int pos_y=rand()%(image.height()-size_y-1)+size_y;


          //  cout <<pos_x<<" "<< pos_y<< " "<< size_x<<" "<<size_y<< " "<< image.height()<< " "<< image.width() <<endl;

            //D+A-B-C
            double filtered_total_sum=integral_image(pos_x,pos_y,0,0)+integral_image(pos_x-size_x,pos_y-size_y,0,0)
                                -integral_image(pos_x-size_x,pos_y,0,0)-integral_image(pos_x,pos_y-size_y,0,0);



            double feature_value=0;
            if (which_feature==2||which_feature==3)
            {
                feature_value=2*(filtered_total_sum)/3;
            }
            else
            {
                feature_value=filtered_total_sum/2;
            }
            feature_vector.push_back(feature_value);
        }

     //   cout<<"Haar Done"<<endl;

        return feature_vector;
    }

    vector<double> extract_features(const string &filename)
    {
        // For Color, Make it 3
      //  cout<< filename<<endl;
        CImg<double> image(filename.c_str());
        CImg<double> gray_image(image.width(),image.height(),1,1);
        vector<double>  feature_vector;
        int grayscale=1;  // Test with Grayscale and Color
        if (feature_type==2)
        {
            feature_vector=haar_features(image);
        }
	else if (feature_type==4)
        {
            feature_vector=deep_features(filename);
        }
        else
        {
            if(image.spectrum() == 3 && grayscale==1)
            {
                gray_image= image.get_RGBtoHSI().get_channel(2);
                image=gray_image;
            }

          //  image=gray_image;
            image.resize(size,size,1,grayscale);
            image.normalize(0,255).save("test.jpg");
            for(int i=0;i<image.width();i++)
            {
                for(int j=0;j<image.height();j++)
                {
                    for(int c=0;c<grayscale;c++)
                    {
                        feature_vector.push_back(normalize(image(i,j,0,c)));
                    }

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
    bool fexists(const char *filename) {
        ifstream ifile(filename);
        return ifile;
    }
    static const int size=80;  // subsampled image resolution

    map<string, CImg<double> > models; // trained models
    vector<double> classes;
};
