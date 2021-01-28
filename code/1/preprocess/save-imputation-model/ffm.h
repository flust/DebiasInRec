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



using namespace std;

typedef double ImpFloat;
typedef double ImpDouble;
typedef unsigned int ImpInt;
typedef unsigned long int ImpLong;
typedef vector<ImpDouble> Vec;

const int MIN_Z = -1000;

class Parameter {
public:
    ImpFloat omega, omega_neg, lambda, r;
    ImpInt nr_pass, k, nr_threads;
    string model_path, predict_path;
    bool self_side, freq, item_weight;
    Parameter():omega(0), omega_neg(1), lambda(4), r(-1), nr_pass(20), k(4), nr_threads(1), self_side(true), freq(false), item_weight(false) {};
};

class Node {
public:
    ImpInt fid;
    ImpLong idx;
    ImpDouble val;
    Node(): fid(0), idx(0), val(0) {};
};

class YNode {
public:
    ImpDouble fid;
    ImpLong idx;
    ImpDouble val;
    ImpDouble val_imp;
    ImpDouble expyy;
    ImpDouble delta;
    YNode(): fid(0), idx(0), val(0), val_imp(0) {};
};

class ImpData {
public:
    string file_name;
    ImpLong m, n, nnz_x, nnz_y;
    ImpInt f;
    vector<Node> N;
    vector<YNode> M;
    vector<Node*> X;
    vector<YNode*> Y;


    vector<vector<Node>> Ns;
    vector<vector<Node*>> Xs;
    vector<ImpLong> Ds;
    vector<vector<ImpLong>> freq;


    ImpData(string file_name): file_name(file_name), m(0), n(0), f(0) {};
    void read(bool has_label, const ImpLong* ds=nullptr);
    void print_data_info();
    void split_fields();
    void transY(const vector<YNode*> &YT);
};


class ImpProblem {
public:
    ImpProblem(shared_ptr<ImpData> &U, shared_ptr<ImpData> &Uva,
            shared_ptr<ImpData> &V, shared_ptr<Parameter> &param)
        :U(U), Uva(Uva), V(V), param(param) {};

    void load_imputation_model(string & model_imp_path);
    void init();
    void solve();
    void save_model(string & model_path);
    void save_Pva_Qva(string & model_path);
    ImpDouble func();
    
    void write_header(ofstream& o_f) const;
    void write_W_and_H(ofstream& o_f) const;
private:
    ImpDouble loss, lambda, w, wn, tr_loss;

    shared_ptr<ImpData> U, Uva, V;
    shared_ptr<Parameter> param;

    ImpInt k, fu, fv, f;
    ImpLong m, n;
    ImpLong mt;

    vector<Vec> W, H, P, Q, Pva, Qva;
    Vec at, bt;
    Vec a, b, va_loss_prec, va_loss_ndcg, sa, sb, Gneg, item_w;
    ImpDouble gauc=0, gauc_all=0;
    ImpDouble auc = 0;
    ImpDouble L_pos; 
    vector<ImpInt> top_k;
    vector<Vec> P_imp, Q_imp;
    ImpInt k_imp, fu_imp, fv_imp, f_imp;

    void init_pair(const ImpInt &f12, const ImpInt &fi, const ImpInt &fj,
            const shared_ptr<ImpData> &d1, const shared_ptr<ImpData> &d2);
    void init_y_imp();

    void add_side(const Vec &p, const Vec &q, const ImpLong &m1, Vec &a1);
    void calc_side();
    void init_item_weights();
    void init_y_tilde();
    void init_L_pos();
    void init_expyy();
    ImpDouble calc_cross(const ImpLong &i, const ImpLong &j);

    void update_side(const bool &sub_type, const Vec &S, const Vec &Q1, Vec &W1, const vector<Node*> &X12, Vec &P1);
    void update_cross(const bool &sub_type, const Vec &S, const Vec &Q1, Vec &W1, const vector<Node*> &X12, Vec &P1);

    void UTx(const Node *x0, const Node* x1, const Vec &A, ImpDouble *c);
    void UTX(const vector<Node*> &X, ImpLong m1, const Vec &A, Vec &C);
    void QTQ(const Vec &C, const ImpLong &l);

    ImpDouble pq(const ImpInt &i, const ImpInt &j,const ImpInt &f1, const ImpInt &f2);
    ImpDouble norm_block(const ImpInt &f1,const ImpInt &f2);

    ImpDouble l_pos_grad(const YNode* y, const ImpDouble iw);
    ImpDouble l_pos_hessian(const YNode* y, const ImpDouble iw);

    void solve_cross(const ImpInt &f1, const ImpInt &f2);
    void gd_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, Vec &G);
    void gd_pos_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, Vec &G);
    void gd_neg_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, Vec &G);
    void hs_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V, const Vec &VQTQ, Vec &Hv, const Vec &Q1, const vector<Node*> &UX, const vector<YNode*> &Y, Vec &Hv_);
    void hs_pos_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V, const Vec &VQTQ, Vec &Hv, const Vec &Q1, const vector<Node*> &UX, const vector<YNode*> &Y, Vec &Hv_);
    void hs_neg_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V, const Vec &VQTQ, Vec &Hv, const Vec &Q1, const vector<Node*> &UX, const vector<YNode*> &Y, Vec &Hv_);

    void cg(const ImpInt &f1, const ImpInt &f2, Vec &W1, const Vec &Q1, const Vec &G, Vec &P1);
void line_search(const ImpInt &f1, const ImpInt &f2, Vec &S1, const Vec &Q1, const Vec &W1, Vec &P1, const Vec &G);

    void calc_delta_y_side(vector<YNode*> &Y, const ImpLong m1, const Vec &XS, const Vec &Q);
    void calc_delta_y_cross(vector<YNode*> &Y, const ImpLong m1, const Vec &XS, const Vec &Q);
    ImpDouble calc_L_pos(vector<YNode*> &Y, const ImpLong m1, const ImpDouble theta);

    void cache_sasb();


    void one_epoch();
    void init_va(ImpInt size);

    void init_Pva_Qva_at_bt();
    void pred_z(const ImpLong i, ImpDouble *z);
    ImpDouble pred_i_j(const ImpLong i, const ImpLong j);
    void ndcg(ImpDouble *z, ImpLong i, vector<ImpDouble> &hit_counts);
    void pred_items();
    ImpDouble calc_gauc_i(Vec &z, ImpLong i, bool all);
    ImpDouble calc_auc_i(Vec &z, Vec &label);
    void prec_k(ImpDouble *z, ImpLong i, vector<ImpInt> &top_k, vector<ImpLong> &hit_counts);
    void validate();
    void calc_gauc();
    void calc_auc();
    void logloss();
    void print_epoch_info(ImpInt t);
};

