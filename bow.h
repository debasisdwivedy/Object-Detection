//
// Created by Charlene Tay on 3/24/16.
//

#ifndef CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H
#define CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H

#endif //CTAY_DDWIVEDY_TOUAHMED_A3_BOW_H

#define num_clusters 500
static const int imgsize = 128;

class Histograms
{
public:
    string filename;
    int category;
    int arr[num_clusters];
    Histograms()
    {
        for(int i=0;i<num_clusters;i++)
        {
            arr[i]=0;
        }
        filename="";
        category=0;
    }

};

class BOW : public Classifier
{
public:
    BOW(const vector <string> &_class_list) : Classifier(_class_list) { }

    virtual void train(const Dataset &filenames)
    {

        // Set up file containing all the SIFT descriptors for all the training images.
        ofstream datafile;
        datafile.open("bow_train.dat");

        // For each folder of images
        for (Dataset::const_iterator c_iter = filenames.begin(); c_iter != filenames.end(); ++c_iter)
        {
            cout << "BOW: Processing " << c_iter->first << endl;
            int target = distance(filenames.begin(), c_iter) + 1;

            // For each image in the folder
            for (int i = 0; i < c_iter->second.size(); i++)
            {
                // Load each image, convert to grayscale and compute SIFT descriptors
                CImg<double> input_image(c_iter->second[i].c_str());
                CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
                // resize image.
                gray.resize(imgsize,imgsize,1,1);
                vector <SiftDescriptor> descriptors = Sift::compute_sift(gray);

                // Record values of each SIFT descriptor
                for (int j = 0; j < descriptors.size(); j++)
                {
                    vector<float> v = descriptors[j].descriptor;
                    datafile << target << "_" << c_iter->second[i].c_str() << " ";
                    for (int k = 0; k < v.size(); k++)
                    {
                        datafile << (k + 1) << ":" << descriptors[j].descriptor[k] << " ";
                    }

                    datafile << "\n";

                }

                datafile << "\n";

            }

        }
        datafile.close();


        // Run Yakmo to cluster descriptors. 100 is number of clusters. Change this value if changing k (number of clusters).
        // bow_train.dat is the data file containing all the SIFT descriptors. Each line is 1 descriptor.
        // Each line in the file starts with the class value followed by each value in the 128d SIFT vector.
        // kmeans_centroid.txt is the file that yakmo saves the cluster centroid values to.
        // kmeans_output.txt is the output file - each line represents the image and the cluster that each SIFT descriptor corresponds to.
        system("yakmo -k 500 bow_train.dat kmeans_centroid.txt - -O 2 >kmeans_output.txt");

        // Read in kmeans_output.txt and create a histogram for each image
        ifstream outputfile("kmeans_output.txt");
        string str;
        vector<Histograms*> train_hist;
        Histograms *hist=new Histograms();
        while (std::getline(outputfile, str))
        {

            if(str.substr(0,1)==" ")
            {
                train_hist.push_back(hist);
                hist=new Histograms();
            }

            else
            {
                int index=str.find(" ");
                string name=str.substr(0,index);
                string s=str.substr(index+1);
                int value=atoi(s.c_str());

                index=name.find("_");
                string p=name.substr(0,index);
                int category=atoi(p.c_str());

                string file_path=name.substr(index+1);

               // cout << value << " " << category << " " << file_path << endl;
                hist->category=category;
                hist->filename=file_path;
                hist->arr[value-1]+=1;
            }

        }
        //cout<<train_hist.size()<<endl;

        ofstream bowhist;
        bowhist.open("bow_histograms.dat");

        for(std::vector<Histograms*>::iterator it = train_hist.begin(); it != train_hist.end(); ++it)
        {

            bowhist<<(*it)->category<<" ";
            for(int j=0;j<500;j++)
            {
                //cout<<j+1<<": "<<(*it)->arr[j]<<" ";
                bowhist<<j+1<<":"<<(*it)->arr[j]<<" ";
            }
            //cout << endl;
            bowhist<<"\n";
        }
        bowhist.close();

        // Train SVM with bow_histograms.dat file
        chdir("svm_multiclass/");
        system("make > garbage.txt ");
        system("./svm_multiclass_learn -c 50 ../bow_histograms.dat ../bow_svm_model >../bow_train.txt");
        chdir("../");

    }


    virtual string classify(const string &filename)
    {
        // Create data file containing all the SIFT descriptors across all test images.
        ofstream datafile;
        datafile.open("bow_test.dat");

        int target = 1;

            // Load each test image, convert to grayscale and compute SIFT descriptors.
            CImg<double> input_image(filename.c_str());
            CImg<double> gray = input_image.get_RGBtoHSI().get_channel(2);
            vector <SiftDescriptor> descriptors = Sift::compute_sift(gray);

            for (int j = 0; j < descriptors.size(); j++)
            {
                vector<float> v = descriptors[j].descriptor;
                datafile << target << "_" << filename << " ";
                for (int k = 0; k < v.size(); k++)
                {
                    datafile << (k + 1) << ":" << descriptors[j].descriptor[k] << " ";
                }
                datafile << "\n";

            }

        datafile.close();

        // Run Yakmo to cluster SIFT descriptors from the test images using the centroids obtained from training data.
        // kmeans_centroid.txt is the input file (generated from training step) containing SIFT values of each cluster centroid.
        // bow_test.txt is the output file - each line represents the image and the cluster that each SIFT descriptor corresponds to.
        system("yakmo - kmeans_centroid.txt bow_test.dat -O 2 >bow_test.txt");

        // Read in bow_test.txt file and create histogram for each image.
        ifstream outputfile("bow_test.txt");
        string str;
        vector<Histograms*> train_hist;
        Histograms *hist=new Histograms();
        while (std::getline(outputfile, str))
        {

            if(str.substr(0,1)==" ")
            {
                train_hist.push_back(hist);
                hist=new Histograms();
            }

            else
            {
                int index=str.find(" ");
                string name=str.substr(0,index);
                string r=str.substr(index+1);
                int value=atoi(r.c_str());

                index=name.find("_");
                string t=name.substr(0,index);
                int category=atoi(t.c_str());

                string file_path=name.substr(index+1);

                //cout << value << " " << category << " " << file_path << endl;
                hist->category=category;
                hist->filename=file_path;
                hist->arr[value-1]+=1;
            }

        }
        //train_hist.push_back(hist);
        //cout<<train_hist.size()<<endl;

        ofstream bowhist;
        bowhist.open("bow_test_histograms.dat");


        bowhist<<hist->category<<" ";
        for(int j=0;j<500;j++)
        {
            //cout<<j+1<<": "<<(*it)->arr[j]<<" ";
            bowhist<<j+1<<":"<<hist->arr[j]<<" ";
        }
        //cout << endl;
        bowhist<<"\n";

        bowhist.close();

        // Classify images using SVM_Multiclass
        chdir("svm_multiclass/");
        system("make > garbage.txt");
        system("./svm_multiclass_classify ../bow_test_histograms.dat ../bow_svm_model ../bow_test_output.txt >../test.txt ");
        chdir ("../");

        string str1;
        string output_label;
        ifstream testoutputfile("bow_test_output.txt");
        while (std::getline(testoutputfile, str1))
        {
            int index=str1.find(" ");
            string y=str1.substr(0,index);
            int c_index=atoi(y.c_str());
            output_label=class_list[c_index-1];
            cout<<output_label<<endl;
        }
        testoutputfile.close();
        return output_label;

    }

    virtual void load_model()
    {
        return;

    }
};

