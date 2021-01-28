#include <iostream>
#include <random>
#include <fstream>
#include <sstream>
#include <memory>
#include <cstring>
#include <stdlib.h>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <climits>
#include <utility>
#include <numeric>
#include <cassert>


#include <immintrin.h>

#include<omp.h>
#include "mkl.h"

#include "./npy.hpp"


using namespace std;

typedef double ImpFloat;
typedef double ImpDouble;
typedef unsigned int ImpInt;
typedef unsigned long int ImpLong;
typedef vector<ImpDouble> Vec;

const int MIN_Z = -1000;

const int PIVOT_I = 0;
const int PIVOT_J = 1;
const int PIVOT_K = 2;

const int POINT_R = -1;
const int GLOBAL_R = 0;
const int POS_R = 1;

class Parameter {
public:
    ImpFloat omega, lambda, lambda_imp;
    ImpInt nr_pass, nr_pass_imp, d, d_imp, nr_threads;
    string model_path, predict_path;
    Parameter():omega(0), lambda(4), lambda_imp(4), nr_pass(20), nr_pass_imp(20), d(4), d_imp(4), nr_threads(1){};
};

class Node {
public:
    ImpLong idx;
    ImpDouble val;
    Node(): idx(0), val(0) {};
};

class YNode {
public:
    ImpDouble label, val, ival;
    ImpLong idx, jdx, kdx;

    ImpDouble expyy;
    ImpDouble delta;

    YNode(): label(0), val(0), ival(0), idx(0), jdx(0), kdx(0), expyy(0), delta(0){};
};

class ImpData {
public:
    string file_name;
    ImpInt pivot, K;
    ImpLong m, nnz_x, nnz_y, D;

    vector<YNode> M;
    vector<Node> N;

    vector<YNode*> Y;
    vector<Node*> X;

    ImpData(int pivot):pivot(pivot), K(0),  m(0), nnz_x(0), nnz_y(0), D(0){};
    ImpData(string file_name, int pivot): file_name(file_name), pivot(pivot), K(0), m(0), nnz_x(0), nnz_y(0), D(0) {};
    void read(bool has_label, const ImpLong Ds=0);
    void decode_onehot();
    void transY(const vector<YNode*> &YT);
    void dummy_position();
    void normalize();
};


class ImpProblem {
public:
    ImpProblem(shared_ptr<ImpData> &U, shared_ptr<ImpData> &Uva,
            shared_ptr<ImpData> &V, shared_ptr<ImpData> &P,
            const ImpDouble &lambda, const ImpDouble &w,
            const ImpInt &d, const ImpInt &nr_threads, const ImpInt &nr_pass, const bool solve_imp)
        :U(U), Uva(Uva), V(V), P(P), lambda(lambda), w(w),
        d(d), nr_threads(nr_threads), nr_pass(nr_pass), solve_imp(solve_imp){
        
            m = U->m;
            n = V->m;
            K = U->K;

        };
    
    ImpProblem(shared_ptr<ImpData> &U, shared_ptr<ImpData> &Uva,
            shared_ptr<ImpData> &V,
            const ImpDouble &lambda, const ImpDouble &w,
            const ImpInt &d, const ImpInt &nr_threads, const ImpInt &nr_pass, const bool solve_imp)
        :U(U), Uva(Uva), V(V), P(NULL), lambda(lambda), w(w),
        d(d), nr_threads(nr_threads), nr_pass(nr_pass), solve_imp(solve_imp){
        
            m = U->m;
            n = V->m;
            K = U->K;

        };


    Vec iW, iH, iZ;
    Vec E_u, E_v, E_p, W, H, Z, W_va;
    Vec imp_r;

    bool do_quiet = false;

    void init();
    void solve();
    void input_whz(const shared_ptr<ImpData> &U1, Vec &iW1, Vec &iH1, Vec &iZ1);
    void init_imp_r_whz(const Vec &imp_r);
    void calc_imp_r();
    void calc_single_imp_r();
    void save_npy_files();

private:
    shared_ptr<ImpData> U, Uva, V, P;

    ImpDouble loss, lambda, w, tr_loss;

    ImpInt d, nr_threads, nr_pass;
    ImpLong m, n, K;
    ImpLong mt;

    Vec Gneg, HTHZTZ, iHTHiZTZ;

    ImpDouble gauc=0, gauc_all=0;
    ImpDouble auc = 0;
    ImpDouble L_pos;

    bool solve_imp = false;


    void calc_val_expyy(const shared_ptr<ImpData> &U1);
    void init_ival(const shared_ptr<ImpData> &U1);

    void init_params();
    void init_y_tilde_expyy();
    void init_L_pos();

    ImpDouble calc_cross(const ImpLong &i, const ImpLong &j, const ImpLong &k);

    void update_cross(const shared_ptr<ImpData> U1, const Vec &S, Vec &E, Vec &W1);

    void UTx(const Node *x0, const Node* x1, const Vec &A, ImpDouble *c);
    void UTX(const vector<Node*> &X, ImpLong m1, const Vec &A, Vec &C);
    void QTQ(const Vec &C, const ImpLong &l);


    ImpDouble l_pos_grad(const YNode* y);
    ImpDouble l_pos_hessian(const YNode* y);


    void solve_block(const shared_ptr<ImpData> &U1, Vec &W1, const Vec &H1,
            const Vec &Z1, const Vec &iW1, const Vec &iH1, const Vec &iZ1, Vec &E1);

    void gd_cross(const shared_ptr<ImpData> &U1, const Vec &W1, const Vec &H1,
            const Vec &Z1, const Vec &iW1, const Vec &iH1, const Vec &iZ1, const Vec &E1, Vec &G);

    void gd_neg_cross(const shared_ptr<ImpData> &U1, const Vec &W1,
            const Vec &H1, const Vec &Z1, const Vec &iW1, const Vec &iH1, const Vec &iZ1, Vec &G);

    void gd_pos_cross(const shared_ptr<ImpData> &U1, const Vec &H1, const Vec &Z1, Vec &G);

    void hs_cross(const shared_ptr<ImpData> &U1, Vec &M,const Vec &H1, const Vec &Z1, Vec &Hv, Vec &Hv_);

    void hs_pos_cross(const shared_ptr<ImpData> U1, Vec &M,
            const Vec H1, const Vec Z1, Vec &Hv, Vec &Hv_);

    void hs_neg_cross(const shared_ptr<ImpData> U1, Vec &M, Vec &Hv, Vec &Hv_);

    void cg(const shared_ptr<ImpData> &U1, const Vec &G1, const Vec &H1, const Vec &Z1, Vec &S1);
    void line_search(const shared_ptr<ImpData> &U1, const Vec &E, const Vec &H1,
        const Vec &Z1, const Vec &G, Vec &S1);

    void calc_delta_y_cross(const shared_ptr<ImpData> &U1, const Vec &XS,
            const Vec &H1, const Vec &Z1);

    void calc_delta_y_pos(vector<YNode*> &Y, const ImpLong m1, const Vec &S);

    ImpDouble calc_L_pos(const shared_ptr<ImpData>  &U1, const ImpDouble theta);


    void one_epoch();

    void init_va();
    ImpDouble calc_gauc_i(const Vec &z, const ImpLong &i, bool all);
    ImpDouble calc_auc_i(const Vec &z, const Vec &label);
    void calc_gauc();
    void calc_auc();
    void logloss();
    void update_Pva_Yva();
    void print_epoch_info(ImpInt t);
    void echo_pos_avg(const ImpLong &j);


};

