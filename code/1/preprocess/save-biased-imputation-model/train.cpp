#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ffm.h"

struct Option {
    shared_ptr<Parameter> param;
    string xc_path, xt_path, tr_path, te_path, model_imp, save_imp;
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
    "-imp <path>: set path to imputation model\n"
    "-save-imp <path>: save imputation model\n"
    "-o <path>: set path to save model file\n"
    "-w <omega>: set cost weight for the unobserves\n"
    "-wn <omega>: set cost weight for the negatives\n"
    "-r <rating>: set rating for the negatives\n"
    "-c <threads>: set number of cores\n"
    "-k <rank>: set number of rank\n"
    "--no-item: set item-bias\n"
    "--freq: enable freq-aware lambda\n"
    "--weighted: enable weighted logloss\n"
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
        else if(args[i].compare("-wn") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -t");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-wn should be followed by a number");
            option.param->omega_neg = atof(argv[i]);
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
        else if(args[i].compare("-imp") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -imp");
            i++;

            option.model_imp = string(args[i]);
        }
        else if(args[i].compare("-save-imp") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -save-imp");
            i++;

            option.save_imp = string(args[i]);
        }
        else if(args[i].compare("--ns") == 0)
        {
            option.param->self_side = false;
        }
        else if(args[i].compare("--freq") == 0){
            option.param->freq = true;
        }
        else if(args[i].compare("--weighted") == 0){
            option.param->item_weight = true;
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


        shared_ptr<ImpData> U = make_shared<ImpData>(option.tr_path);
        shared_ptr<ImpData> V = make_shared<ImpData>(option.xt_path);

        shared_ptr<ImpData> Ut = make_shared<ImpData>(option.te_path);


        U->read(true);
        U->split_fields();

        V->read(false);
        V->transY(U->Y);
        V->split_fields();

        //assert(U->n == V->m);

        if (!Ut->file_name.empty()) {
            Ut->read(true, U->Ds.data());
            Ut->split_fields();
        }

        ImpProblem prob(U, Ut, V, option.param);
        if( !option.model_imp.empty() )
            prob.load_imputation_model(option.model_imp);
        prob.init();
        prob.solve();
        if( !option.param->model_path.empty() )
          prob.save_model(option.param->model_path );
        if( !option.save_imp.empty() )
          prob.save_Pva_Qva(option.save_imp);
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}

