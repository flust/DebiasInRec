#include <iostream>
#include <cstring>
#include <stdexcept>

#include "mf.h"

#include <fenv.h>


struct Option
{
    shared_ptr<Parameter> param;
    ImpInt verbose;
    string test_path;
    bool test;
};

string predict_help()
{
    return string(
    "usage: predict [option] model_path"
    "\n"
    "options:\n"
    "-p <path>: set path to test set\n"
    "-r <path>: set path to save result\n"
    );
}


Option parse_option(int argc, char **argv)
{
    vector<string> args;
    for(int i = 0; i < argc; i++)
        args.push_back(string(argv[i]));

    if(argc == 1)
        throw invalid_argument(predict_help());

    Option option;
    option.verbose = 1;
    option.test = false;
    option.param = make_shared<Parameter>();
    int i = 0;
    for(i = 1; i < argc; i++)
    {
        if(args[i].compare("-p") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -p");
            i++;
            option.test = true;
            option.test_path = string(args[i]);
        }
        else if(args[i].compare("-r") == 0)
        {
            if(i == argc-1)
                throw invalid_argument("need to specify path after -r");
            i++;

            option.param->predict_path = string(args[i]);
        }
        else
        {
            break;
        }
    }

    if (i >= argc)
        throw invalid_argument("model or precdict path missing");
    option.param->model_path = string(args[i++]);

    return option;
}

int main(int argc, char **argv)
{
    try
    {
        Option option = parse_option(argc, argv);
        

        shared_ptr<ImpData> data = make_shared<ImpData>("FakeData");
        shared_ptr<ImpData> test_data = make_shared<ImpData>(option.test_path);
        if (option.test)
        {
            test_data->read();
        }
        ImpProblem prob(data, option.param);
        prob.initialize();
        prob.load();
    }
    catch (invalid_argument &e)
    {
        cerr << e.what() << endl;
        return 1;
    }
    return 0;
}
