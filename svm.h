//
// Created by Tousif Ahmed on 3/10/16.
//

#ifndef CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H
#define CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H

#endif //CTAY_DDWIVEDY_TOUAHMED_A3_SVM_H

class Integral_Image
{
public:
    CImg<double> ii;
    Integral_Image(CImg<double> i){ii=i;}

};


class SVM : public Classifier
{
    int feature_type;
    string svm_model_name;
public:
    SVM(const vector<string> &_class_list,int feature) : Classifier(_class_list)
    {
        feature_type=feature;
        svm_model_name="baseline_svm_model";
    }

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


        string svm_command="./svm_multiclass_learn -c 50 ../train.dat ../"+svm_model_name+" >../train.txt";
      //  system("./svm_multiclass_learn -c 50 ../train.dat ../svm_model >../train.txt");
        system(svm_command.c_str());
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

        string svm_command="./svm_multiclass__classify ../test.dat ../"+svm_model_name+" ../test_output.txt >../test.txt ";

      //  system("./svm_multiclass_classify ../test.dat ../svm_model ../test_output.txt >../test.txt ");
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

protected:
    // extract features from an image, which in this case just involves resampling and
    // rearranging into a vector of pixel data.

    CImg<double> averageGrayscale(CImg<double> input){
        //Creates a new grayscale image with same size as the input initialized to all 0s (black)

        //FIXME fill in new grayscale image

        int x, y, col;
        double sum;
        int input_w = input.width();
        int input_h = input.height();
        CImg<double> output(input_w, input_h, 1, 1);

        //FIXME fill in new grayscale image
        for(x = 0; x < input_w; x++) {
            for(y = 0; y < input_h; y++) {
                sum = 0.0;
                sum += input(x, y, 0, 0) * 0.21;
                sum += input(x, y, 0, 1) * 0.71;
                sum += input(x, y, 0, 2) * 0.07;
                output(x,y,0,0) = sum;
            }
        }
        return output;
    }

    vector<double> deep_features(CImg<double> image)
    {
        vector<double>  feature_vector;
        int len=270;
        image.resize(len,len,1,3);
        image.save("resized.jpg");
       // cout<< filename<<endl;
        string command="./overfeat/bin/linux_64/overfeat -f resized.jpg > deep_features.txt";
       // cout<<command<<endl;
	    system(command.c_str());
        ifstream featurefile("deep_features.txt");
        string str;
        getline(featurefile, str);

        //cout<<str<<endl;



        int index=str.find(" ");
        string s=str.substr(0,index);
        string rest=str.substr(index+1);
        int h_index=rest.find(" ");
        int n=atoi(s.c_str());

        s=rest.substr(0,h_index);
        int h=atoi(s.c_str());
        s=rest.substr(h_index+1);
        int r=atoi(s.c_str());

        getline(featurefile,str);
        std::stringstream ss(str);

        double value;
        while (ss >> value)
        {
           feature_vector.push_back(value);


            if (ss.peek() == ' ')
                ss.ignore();
        }

       // cout<<feature_vector.size()<<endl;

       // getline(featurefile,str);
     //   cout<<n <<" h: "<<h <<" w: "<<r<<endl;
        featurefile.close();

        svm_model_name="deep_svm_model";
       return feature_vector; 
        
    }

    double get_haar_feature(CImg<double> image,CImg<double> integral_image)
    {
        int features=5;
        int feature[5][2] = {{2,1}, {1,2}, {3,1}, {1,3}, {2,2}};
        int max_size=24;
        int no_of_features=1000;

        int which_feature = rand() % 5;
        int size_x = feature[which_feature][0];
        int size_y = feature[which_feature][1];

        int max=size_x>size_y?size_x:size_y;

        int r_size_x=rand()%(max_size/size_x)+1;
        int r_size_y=rand()%(max_size/size_y)+1;

        size_x=size_x*r_size_x;
        size_y=size_y*r_size_y;
        int black=rand()%2; // 0 black, 1 white

        int m=black==0?-1:1;

        int pos_x=rand()%(image.width()-size_x-1)+size_x;
        int pos_y=rand()%(image.height()-size_y-1)+size_y;


       //   cout <<pos_x<<" "<< pos_y<< " "<< size_x<<" "<<size_y<< " "<< image.height()<< " "<< image.width() <<endl;

        //D+A-B-C

        int changed_x=pos_x-size_x;
        int changed_y=pos_y-size_y;
        double filtered_total_sum=0;
        if(pos_x>0 && pos_y>0 && changed_x>0 && changed_y>0)
            filtered_total_sum=integral_image(pos_x,pos_y,0,0)+integral_image(pos_x-size_x,pos_y-size_y,0,0)
                                  -integral_image(pos_x-size_x,pos_y,0,0)-integral_image(pos_x,pos_y-size_y,0,0);


        double feature_value=0;
        if (which_feature==2||which_feature==3)
        {
            feature_value=2*m*(filtered_total_sum)/3;
        }
        else
        {
            feature_value=m*filtered_total_sum/2;
        }

        return feature_value;
    }

    vector<double> haar_features(CImg<double> image)
    {
        // CIMG initialize
        CImg<double> gray_image(image.width(),image.height(),1,1);
        CImg<double> integral_image(image.width(),image.height(),1,1);
        CImg<double> row_sum(image.width(),image.height(),1,1);

     //   double integral_image[image.width()][image.height()];
        vector<Integral_Image> integral_images;


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
            gray_image= averageGrayscale(image);//image.get_RGBtoHSI().get_channel(2);
            image=gray_image;
        }
        //image.normalize(0,255).save("test.jpg");
        int haar_size=80;
        image.resize(haar_size,haar_size,1,grayscale);


      //  double integral_image[grayscale][image.width()][image.height()];

        for(int c=0;c<grayscale;c++)
        {
            for(int x=0;x<image.width();x++)
            {
                for(int y=0;y<image.height();y++)
                {

                    row_sum(x,y,0,0)=y>0?row_sum(x,y-1,0,0)+image(x,y,0,0):image(x,y,0,0);
                     integral_image(x,y,0,0)=x>0?row_sum(x,y,0,0)+integral_image(x-1,y,0,0):row_sum(x,y,0,0);
                  //  integral_image[c][x][y]=x>0?row_sum(x,y,0,0)+integral_image[c][x-1][y]:row_sum(x,y,0,0);
                }
            }
            Integral_Image ii(integral_image);
            integral_images.push_back(ii);

        }



       // integral_images.push_back(integral_image);
        // Feature Generation
       for(int i=0;i<no_of_features;i++)
        {
            for(int c=0;c<grayscale;c++)
            {
                integral_image=integral_images.at(c).ii;
                int feature_value=get_haar_feature(image,integral_image);
                feature_vector.push_back(feature_value);

            }

     /*       int which_feature = rand() % 5;
            int size_x = feature[which_feature][0];
            int size_y = feature[which_feature][1];

            int max=size_x>size_y?size_x:size_y;

            int r_size_x=rand()%(max_size/size_x)+1;
            int r_size_y=rand()%(max_size/size_y)+1;

            size_x=size_x*r_size_x;
            size_y=size_y*r_size_y;
            int black=rand()%2; // 0 black, 1 white

            int m=black==0?-1:1;

            int pos_x=rand()%(image.width()-size_x-1)+size_x;
            int pos_y=rand()%(image.height()-size_y-1)+size_y;


          //  cout <<pos_x<<" "<< pos_y<< " "<< size_x<<" "<<size_y<< " "<< image.height()<< " "<< image.width() <<endl;

            //D+A-B-C
            double filtered_total_sum=integral_image(pos_x,pos_y,0,0)+integral_image(pos_x-size_x,pos_y-size_y,0,0)
                                -integral_image(pos_x-size_x,pos_y,0,0)-integral_image(pos_x,pos_y-size_y,0,0);



            double feature_value=0;
            if (which_feature==2||which_feature==3)
            {
                feature_value=2*m*(filtered_total_sum)/3;
            }
            else
            {
                feature_value=m*filtered_total_sum/2;
            }*/
           // feature_vector.push_back(feature_value);
        }

     //   cout<<"Haar Done"<<endl;
        svm_model_name="haar_svm_model";
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
            feature_vector=deep_features(image);
        }
        else
        {
            if(image.spectrum() == 3 && grayscale==1)
            {
               // gray_image= image.get_RGBtoHSI().get_channel(2);
                gray_image=averageGrayscale(image);
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
        //svm_model_name="baseline_svm_model";

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
    static const int size=40;  // subsampled image resolution

    map<string, CImg<double> > models; // trained models
    vector<double> classes;
};
