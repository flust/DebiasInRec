#include <iostream>
#include <cstring>
#include <stdexcept>

#include "ffm.h"

struct Option {
    shared_ptr<Parameter> param;
    string xc_path, xt_path, tr_path, filter_sr_path, te_path, model_path;
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
    "usage: predict-and-filter [options] item_feature_file train_file filter_file model\n"
    "\n"
    "options:\n"
    "-c <threads>: set number of cores\n"
    "-k <rank>: set number of rank\n"
    "--no-item: set item-bias\n"
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
        if(args[i].compare("-c") == 0)
        {
            if((i+1) >= argc)
                throw invalid_argument("missing core numbers after -c");
            i++;
            if(!is_numerical(argv[i]))
                throw invalid_argument("-c should be followed by a number");
            option.param->nr_threads = atof(argv[i]);
        }
        else if(args[i].compare("--ns") == 0)
        {
            option.param->self_side = false;
        }
        else if(args[i].compare("--freq") == 0){
            option.param->freq = true;
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
    option.filter_sr_path = string(args[i++]);
    option.model_path = string(args[i++]);

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
        shared_ptr<ImpData> Ut = make_shared<ImpData>(option.filter_sr_path);

        U->read(true);
        U->split_fields();

        V->read(false);
        V->transY(U->Y);
        V->split_fields();

        Ut->read(true, U->Ds.data());
        Ut->split_fields();

        ImpProblem prob(U, Ut, V, option.param);
        prob.init();
        prob.load_binary_model( option.model_path );
        prob.filter();
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}

