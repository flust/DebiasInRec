#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ffm.h"

struct Option {
    shared_ptr<Parameter> param;
    string xc_path, xt_path, tr_path, te_path;
};

string basename(string path) {
    const char *ptr = strrchr(&*path.begin(), '/');
    if(!ptr)
        ptr = path.c_str();
    else
        ptr++;
    return string(ptr);
}


bool is_numerical(char *str)
{
    int c = 0;
    while(*str != '\0')
    {
        if(isdigit(*str))
            c++;
        str++;
    }
    return c > 0;
}

string train_help()
{
    return string(
    "usage: train [options] item_feature_file train_file\n"
    "\n"
    "options:\n"
    "-l <lambda_2>: set regularization coefficient on r regularizer (default 0.1)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-p <path>: set path to test set\n"
    "-o <path>: set path to save model file\n"
    "-w <omega>: set cost weight for the unobserves\n"
    "-wn <omega>: set cost weight for the negatives\n"
    "-r <rating>: set rating for the negatives\n"
    "-c <threads>: set number of cores\n"
    "-d <rank>: set number of rank\n"
    );
}

Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(train_help());

    Option option;
    option.param = make_shared<Parameter>();
    int i = 0;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-l") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l\
                                        regularization coefficient\
                                        after -l");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-l should be followed by a number");
            option.param->lambda = atof(argv[i]);
        }
        else if(args[i].compare("-d") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify rank after -d");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-d should be followed by a number");
            option.param->d = atoi(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param->nr_pass = atoi(argv[i]);
        }
        else if(args[i].compare("-o") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -o");
            i++;

            option.param->model_path = string(args[i]);
        }
        else if(args[i].compare("-w") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-t should be followed by a number");
            option.param->omega = atof(argv[i]);
        }
        else if(args[i].compare("-r") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-r should be followed by a number");
            option.param->r = atof(argv[i]);
        }
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;

            option.te_path = string(args[i]);
        }
        else
        {
            break;
        }
    }

    if(i >= argc)
        throw invalid_argument("training data not specified");

    option.xt_path = string(args[i++]);
    option.tr_path = string(args[i++]);

    return option;
}

int main(int argc, char *argv[])
{
    try
    {
        Option option = parse_option(argc, argv);
        omp_set_num_threads(option.param->nr_threads);

        shared_ptr<ImpData> U = make_shared<ImpData>(option.tr_path, 0);
        shared_ptr<ImpData> Ut = make_shared<ImpData>(option.te_path,0);

        shared_ptr<ImpData> V = make_shared<ImpData>(option.xt_path, 1);
        shared_ptr<ImpData> P = make_shared<ImpData>(2);

        U->read(true, 0);
        V->read(false, 0);


        V->transY(U->Y);

        P->m = U->K;
        P->D = U->K;
        P->decode_onehot();
        P->transY(U->Y);


        if (!Ut->file_name.empty())
            Ut->read(true, U->D);

        ImpProblem prob(U, Ut, V, P, option.param);
        prob.init();
        prob.solve();
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}

