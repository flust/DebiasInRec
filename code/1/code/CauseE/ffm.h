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
#include <fenv.h>

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
    ImpFloat omega_neg, lambda, ldiff, r;
    ImpInt nr_pass, k, nr_threads;
    string model_path, predict_path;
    bool self_side, freq, item_weight;
    Parameter(): omega_neg(1), lambda(4), ldiff(1), r(0), nr_pass(20), k(4), nr_threads(1), self_side(true), freq(false), item_weight(false) {};
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
    ImpDouble expyy;
    ImpDouble delta;
    YNode(): fid(0), idx(0), val(0) {};
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
    ImpProblem(shared_ptr<ImpData> &U, shared_ptr<ImpData> &U_treat, shared_ptr<ImpData> &Uva,
            shared_ptr<ImpData> &V, shared_ptr<ImpData> &V_treat, shared_ptr<Parameter> &param)
        :U(U), U_treat(U_treat), Uva(Uva), V(V), V_treat(V_treat), param(param) {};

    void init();
    void solve();
    void save_model(string & model_path);
    
    void write_header(ofstream& o_f) const;
    void write_W_and_H(ofstream& o_f) const;
private:
    ImpDouble loss, lambda, ldiff, wn, tr_loss;
    ImpDouble c_norm, t_norm;

    shared_ptr<ImpData> U,  U_treat, Uva, V, V_treat;
    shared_ptr<Parameter> param;

    ImpInt k, fu, fv, f;
    ImpLong m, n;
    ImpLong m_treat, n_treat;
    ImpLong mt;

    vector<Vec> W, H, P, Q, Pva, Qva;
    vector<Vec> W_treat, H_treat, P_treat, Q_treat;
    Vec va_loss_prec, va_loss_ndcg, Gneg, item_w;
    ImpDouble gauc=0, gauc_all=0;
    ImpDouble auc = 0;
    ImpDouble L_pos; 
    ImpDouble L_pos_treat; 
    vector<ImpInt> top_k;

    void init_pair(const ImpInt &f12, const ImpInt &fi, const ImpInt &fj,
            const shared_ptr<ImpData> &d1, const shared_ptr<ImpData> &d2,
            const shared_ptr<ImpData> &d1_treat, const shared_ptr<ImpData> &d2_treat);

    void add_side(const Vec &p, const Vec &q, const ImpLong &m1, Vec &a1);
    void init_item_weights();
    void init_y_tilde();
    void init_L_pos();
    void init_expyy();
    ImpDouble calc_cross(const ImpLong &i, const ImpLong &j);
    ImpDouble calc_cross_treat(const ImpLong &i, const ImpLong &j);

    void update_cross(const bool &sub_type, const Vec &S, const Vec &Q1, Vec &W1, const vector<Node*> &X12, Vec &P1, bool do_ctrl);

    void UTx(const Node *x0, const Node* x1, const Vec &A, ImpDouble *c);
    void UTX(const vector<Node*> &X, ImpLong m1, const Vec &A, Vec &C);
    void QTQ(const Vec &C, const ImpLong &l);

    ImpDouble pq(const ImpInt &i, const ImpInt &j,const ImpInt &f1, const ImpInt &f2);
    ImpDouble norm_block(const ImpInt &f1,const ImpInt &f2);

    ImpDouble l_pos_grad(const YNode* y, const ImpDouble iw);
    ImpDouble l_pos_hessian(const YNode* y, const ImpDouble iw);

    void solve_cross(const ImpInt &f1, const ImpInt &f2);
    void gd_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, const Vec &Wreg, Vec &G, bool do_ctrl);
    void gd_pos_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, Vec &G, bool do_ctrl);
    void hs_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V, Vec &Hv, const Vec &Q1, const vector<Node*> &UX, const vector<YNode*> &Y, Vec &Hv_, const ImpDouble norm_);
    void hs_pos_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V, Vec &Hv, const Vec &Q1, const vector<Node*> &UX, const vector<YNode*> &Y, Vec &Hv_, const ImpDouble norm_);

    void cg(const ImpInt &f1, const ImpInt &f2, Vec &W1, const Vec &Q1, const Vec &G, Vec &P1, bool do_ctrl);
    void line_search(const ImpInt &f1, const ImpInt &f2, Vec &S1, const Vec &Q1, const Vec &W1, const Vec &Wreg, Vec &P1, const Vec &G, bool do_ctrl);

    void calc_delta_y_cross(vector<YNode*> &Y, const ImpLong m1, const Vec &XS, const Vec &Q);
    ImpDouble calc_L_pos(vector<YNode*> &Y, const ImpLong m1, const ImpDouble theta, const ImpDouble norm_);


    void one_epoch();
    void init_va(ImpInt size);

    void init_Pva_Qva_at_bt();
    void init_Pva_Qva_at_bt_treat();
    void pred_z(const ImpLong i, ImpDouble *z);
    ImpDouble pred_i_j(const ImpLong i, const ImpLong j);
    void ndcg(ImpDouble *z, ImpLong i, vector<ImpDouble> &hit_counts);
    void pred_items();
    ImpDouble calc_gauc_i(Vec &z, ImpLong i, bool all);
    ImpDouble calc_auc_i(Vec &z, Vec &label);
    void prec_k(ImpDouble *z, ImpLong i, vector<ImpInt> &top_k, vector<ImpLong> &hit_counts);
    void calc_auc();
    void logloss();
    void logloss_treat();
    void print_epoch_info(ImpInt t);
};

