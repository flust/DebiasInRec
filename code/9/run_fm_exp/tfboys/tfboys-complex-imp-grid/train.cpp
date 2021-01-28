#include <iostream>
#include <cstring>
#include <stdexcept>

#include <fenv.h>

#include "ffm.h"

struct Option {
    shared_ptr<Parameter> param;
    string xc_path, xt_path, tr_path, te_path, imp_path;
    ImpInt do_imp_r = -1;
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
    "usage: train [options] item_feature_file imp_file\n"
    "\n"
    "options:\n"
    "-l <lambda_2>: set regularization coefficient on r regularizer (default 0.1)\n"
    "-t <iter>: set number of iterations (default 20)\n"
    "-p <path>: set path to test set\n"
    "-w <omega>: set cost weight for the unobserves\n"
    "-c <threads>: set number of cores\n"
    "-d <rank>: set number of rank\n"
    "-imp-r do imputation (default -1)\n"
    "\t -1 -- point-wise imputation \n"
    "\t  0 -- global imputation \n"
    "\t  1 -- position-wise imputation \n"
    "--save-model save embedding model"
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
                throw invalid_argument("need to specify L\
                                        regularization coefficient\
                                        after -L");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-L should be followed by a number");
            option.param->lambda_imp = atof(argv[i]);
        }
        else if(args[i].compare("-d") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify rank after -D");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-k should be followed by a number");
            option.param->d_imp = atoi(argv[i]);
        }
        else if(args[i].compare("-t") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -T");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-T should be followed by a number");
            option.param->nr_pass_imp = atoi(argv[i]);
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
        else if(args[i].compare("-imp-r") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("need to specify max number of\
                                        iterations after -imp-r");
            i++;

            if(!is_numerical(argv[i]))
                throw invalid_argument("-impr should be followed by a number");
            option.do_imp_r= atoi(argv[i]);
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
    option.imp_path = string(args[i++]);

    return option;
}

int main(int argc, char *argv[])
{
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT & ~FE_UNDERFLOW);
    try
    {

        Option option = parse_option(argc, argv);
        omp_set_num_threads(option.param->nr_threads);


        cout << "tr_file: "<<  option.tr_path << "\tva: " << option.te_path << "\timp: " << option.imp_path << endl << flush;

        bool solve_imp = true;
        shared_ptr<ImpData> Uimp = make_shared<ImpData>(option.imp_path,0);
        shared_ptr<ImpData> Utimp = make_shared<ImpData>(option.te_path,0);
        shared_ptr<ImpData> Vimp = make_shared<ImpData>(option.xt_path, 1);
        shared_ptr<ImpData> Pimp = make_shared<ImpData>(2);

        Uimp->read(true);
        if (!Utimp->file_name.empty()) {
            Utimp->read(true, Uimp->D);
        }
        Vimp->read(false);

        Vimp->transY(Uimp->Y);
        Pimp->m = Uimp->K;
        Pimp->D = Uimp->K;
        Pimp->decode_onehot();
        Pimp->transY(Uimp->Y);

        ImpProblem imp_prob(Uimp, Utimp, Vimp, Pimp, option.param->lambda_imp, 0,
                option.param->d_imp, option.param->nr_threads, option.param->nr_pass_imp, solve_imp);

        imp_prob.init();
        imp_prob.solve();
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}

