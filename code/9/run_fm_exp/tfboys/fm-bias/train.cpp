#include <iostream>
#include <cstring>
#include <stdexcept>

#include <fenv.h>

#include "ffm.h"

struct Option {
    shared_ptr<Parameter> param;
    string xc_path, xt_path, tr_path, te_path;
    bool save_model = false;
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
    "-w <omega>: set cost weight for the unobserves\n"
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
                throw invalid_argument("-k should be followed by a number");
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
        else if(args[i].compare("-w") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-r should be followed by a number");
            option.param->omega = atof(argv[i]);
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
        else if(args[i].compare("--save-model") == 0)
        {
            option.save_model = true;
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
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW);
    try
    {
        Option option = parse_option(argc, argv);
        omp_set_num_threads(option.param->nr_threads);

        shared_ptr<ImpData> U = make_shared<ImpData>(option.tr_path, 0);
        shared_ptr<ImpData> Ut = make_shared<ImpData>(option.te_path,0);

        shared_ptr<ImpData> V = make_shared<ImpData>(option.xt_path, 1);

        U->read(true);
        V->read(false);

        V->transY(U->Y);

        if (!Ut->file_name.empty()) {
            Ut->read(true, U->D);
        }

        bool is_solve_imp = true;
        ImpProblem imp_prob(U, Ut, V, option.param->lambda, 0,
                option.param->d, option.param->nr_threads, option.param->nr_pass, is_solve_imp);
        imp_prob.init();
        imp_prob.solve();

        is_solve_imp = false;
        ImpProblem pre_prob(U, Ut, V, option.param->lambda,
                option.param->omega, option.param->d,
                option.param->nr_threads, option.param->nr_pass, is_solve_imp);

        pre_prob.input_whz(U, imp_prob.E_u, imp_prob.H, imp_prob.pos_b);
        pre_prob.init();
        pre_prob.solve();
        
        if( option.save_model )
            pre_prob.save_npy_files();
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}

