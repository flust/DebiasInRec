#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ffm.h"

struct Option {
    shared_ptr<Parameter> param;
    string xc_path, xt_path, tr_path, treat_path, te_path;
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
    "-l <lambda_2>: set regularization coefficient on r regularizer (default 4)\n"
    "-ldiff <lambda_2>: set regularization coefficient on diff regularizer (default 1)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-p <path>: set path to test set\n"
    "-c <threads>: set number of cores\n"
    "-k <rank>: set number of rank\n"
    "--treat <path>: set path to treat data\n"
    "--ns: disable self interaction\n"
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
        else if(args[i].compare("-ldiff") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify l\
                                        regularization coefficient\
                                        after -ldiff");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-ldiff should be followed by a number");
            option.param->ldiff = atof(argv[i]);
        }
        else if(args[i].compare("-k") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify rank after -k");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-k should be followed by a number");
            option.param->k = atoi(argv[i]);
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
        else if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("--treat") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after --treat");
            i++;

            option.treat_path = string(args[i]);
        }
        else if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;

            option.te_path = string(args[i]);
        }
        else if(args[i].compare("--ns") == 0)
        {
            option.param->self_side = false;
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
    feenableexcept(FE_INVALID | FE_OVERFLOW);
    try
    {
        Option option = parse_option(argc, argv);
        omp_set_num_threads(option.param->nr_threads);


        shared_ptr<ImpData> U = make_shared<ImpData>(option.tr_path);
        shared_ptr<ImpData> U_treat = make_shared<ImpData>(option.treat_path);
        shared_ptr<ImpData> V = make_shared<ImpData>(option.xt_path);
        shared_ptr<ImpData> V_treat = make_shared<ImpData>(option.xt_path);

        shared_ptr<ImpData> Ut = make_shared<ImpData>(option.te_path);


        U->read(true);
        U->split_fields();
        
        U_treat->read(true);
        U_treat->split_fields();

        V->read(false);
        V->transY(U->Y);
        V->split_fields();

        V_treat->read(false);
        V_treat->transY(U_treat->Y);
        V_treat->split_fields();
        
        assert(U->n == V->m);

        if (!Ut->file_name.empty()) {
            Ut->read(true, U->Ds.data());
            Ut->split_fields();
        }

        ImpProblem prob(U, U_treat, Ut, V, V_treat, option.param);
        prob.init();
        prob.solve();
        if( !option.param->model_path.empty() )
          prob.save_model(option.param->model_path );
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}

