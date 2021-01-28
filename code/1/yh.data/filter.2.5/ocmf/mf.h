#include <vector>
#include <algorithm>
#include <string>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <random>

#include "omp.h"

using namespace std;

#define INF HUGE_VAL
typedef double ImpFloat;
typedef double ImpDouble;
typedef long long int ImpInt;
typedef long long int ImpLong;

class Parameter {

public:
    ImpFloat lambda_u, lambda_i, w, a;
    ImpInt nr_pass, k, nr_threads, scheme;
    string model_path, predict_path;
    bool has_ps;
    Parameter():lambda_u(0.1), lambda_i(0.1), w(1), a(0), nr_pass(20), k(10),nr_threads(1),scheme(0), has_ps(false) {};
};

struct smat {
    vector<ImpLong> row_ptr;
    vector<ImpLong> col_idx;
    vector<ImpDouble> val;
    vector<ImpDouble> p_scores;
};

class ImpData {
public:
    string file_name;
    ImpLong l, m, n;
    ImpLong m_real, n_real;
    smat R;
    smat RT;

    ImpData(string file_name): file_name(file_name), l(0), m(0), n(0) {};
    void read();
    void print_data_info();
    class Compare {
        public:
            const ImpLong *row_idx;
            const ImpLong *col_idx;
            Compare(const ImpLong *row_idx_, const ImpLong *col_idx_) {
                row_idx = row_idx_;
                col_idx = col_idx_;
            }
            bool operator()(size_t x, size_t y) const {
                return  (row_idx[x] < row_idx[y]) || ((row_idx[x] == row_idx[y]) && (col_idx[x]<= col_idx[y]));
            }
    };
};

class ImpProblem {
public:
    shared_ptr<ImpData> data;
    shared_ptr<ImpData> test_data;
    shared_ptr<Parameter> param;
    ImpProblem(shared_ptr<ImpData> &data, shared_ptr<Parameter> &param)
        :data(data), param(param) {};
    ImpProblem(shared_ptr<ImpData> &data, shared_ptr<ImpData> &test_data, shared_ptr<Parameter> &param)
        :data(data), test_data(test_data), param(param) {};

    ImpFloat *W, *H;
    ImpFloat *WT, *HT;

    ImpInt t;
    ImpDouble obj, reg, tr_loss;
    vector<ImpDouble> va_loss;

    ImpFloat start_time;
    ImpFloat U_time, C_time, W_time, H_time, I_time, R_time;

    ImpFloat sum, sq;
    vector<ImpFloat> gamma_w, gamma_h;


    ImpDouble cal_te_loss();
    ImpDouble cal_reg();
    void cal_avg(shared_ptr<ImpData> &d_);

    ImpDouble cal_tr_loss();
    void update(const smat &R, ImpLong i, vector<ImpFloat> &gamma, ImpDouble *u, ImpDouble *v, const ImpFloat lambda, const ImpDouble w_p);
    void save();
    void load();


    void initialize();
    void init_va_loss(ImpInt size);
    void solve();
    void update_R(ImpFloat *wt, ImpFloat *ht, bool add);

    void validate(const vector<ImpInt> &topks);
    void validate_ndcg(const vector<ImpInt> &topks);
    void predict_candidates(const ImpFloat* w, vector<ImpFloat> &Z);
    ImpLong precision_k(vector<ImpFloat> &Z, ImpLong i, const vector<ImpInt> &topks, vector<ImpLong> &hit_counts);
    void init_idcg(const ImpLong ii, vector<ImpDouble> &idcg,const vector<ImpInt> &topks);
    ImpDouble ndcg_k(vector<ImpFloat> &Z, ImpLong i, const vector<ImpInt> &topks, vector<double> &ndcgs);
    
    void cache(ImpDouble* WT, ImpDouble* H, vector<ImpFloat> &gamma, ImpDouble *ut, ImpLong m, ImpLong n);

    void update_coordinates();
    void print_epoch_info();
    void print_header_info(vector<ImpInt> &topks);

    ImpDouble is_hit(const smat &R, ImpLong i, ImpLong argmax);
    void filter_data();
};

