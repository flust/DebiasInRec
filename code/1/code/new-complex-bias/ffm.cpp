#include "ffm.h"

ImpDouble qrsqrt(ImpDouble x)
{
    ImpDouble xhalf = 0.5*x;
    ImpLong i;
    memcpy(&i, &x, sizeof(i));
    i = 0x5fe6eb50c7b537a9 - (i>>1);
    memcpy(&x, &i, sizeof(i));
    x = x*(1.5 - xhalf*x*x);
    return x;
}

ImpDouble sum(const Vec &v) {
    ImpDouble sum = 0;
    for (ImpDouble val: v)
        sum += val;
    return sum;
}

void axpy(const ImpDouble *x, ImpDouble *y, const ImpLong &l, const ImpDouble &lambda) {
    cblas_daxpy(l, lambda, x, 1, y, 1);
}

void scal(ImpDouble *x, const ImpLong &l, const ImpDouble &lambda) {
    cblas_dscal(l, lambda, x, 1);
}

void mm(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpLong n, const ImpInt k) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            l, n, k, 1, a, k, b, n, 0, c, n);
}

void mm(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpLong n, const ImpInt k, const ImpDouble &beta) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            l, n, k, 1, a, k, b, n, beta, c, n);
}

void mm(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong k, const ImpInt l) {
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
            k, k, l, 1, a, k, b, k, 0, c, k);
}

// l * n = (l * k) x (k * n)
void mm(const ImpLong l, const ImpLong n, const ImpLong k, const ImpDouble *a, const ImpDouble *b, ImpDouble *c, const ImpLong a_row, const ImpLong b_row, const ImpLong c_row, const ImpDouble beta){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, l, n, k, 1, a, a_row, b, b_row, beta, c, c_row);
}

// l * n = (l * k) x (k * n), A^T = (l * k)
void mTm(const ImpLong l, const ImpLong n, const ImpLong k, const ImpDouble *a, const ImpDouble *b, ImpDouble *c, const ImpLong a_row, const ImpLong b_row, const ImpLong c_row, const ImpDouble beta){
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, l, n, k, 1, a, a_row, b, b_row, beta, c, c_row);
}

// l * n = (l * k) x (k * n), B^T = (k * n)
void mmT(const ImpLong l, const ImpLong n, const ImpLong k, const ImpDouble *a, const ImpDouble *b, ImpDouble *c, const ImpLong a_row, const ImpLong b_row, const ImpLong c_row, const ImpDouble beta){
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, l, n, k, 1, a, a_row, b, b_row, beta, c, c_row);
}

// l * n = (l * k) x (k * n), A^T = (l * k), B^T = (k * n)
void mTmT(const ImpLong l, const ImpLong n, const ImpLong k, const ImpDouble *a, const ImpDouble *b, ImpDouble *c, const ImpLong a_row, const ImpLong b_row, const ImpLong c_row, const ImpDouble beta){
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, l, n, k, 1, a, a_row, b, b_row, beta, c, c_row);
}

void mv(const ImpDouble *a, const ImpDouble *b, ImpDouble *c,
        const ImpLong l, const ImpInt k, const ImpDouble &beta, bool trans) {
    const CBLAS_TRANSPOSE CBTr= (trans)? CblasTrans: CblasNoTrans;
    cblas_dgemv(CblasRowMajor, CBTr, l, k, 1, a, k, b, 1, beta, c, 1);
}

const ImpInt index_vec(const ImpInt f1, const ImpInt f2, const ImpInt f) {
    return f2 + (f-1)*f1 - f1*(f1-1)/2;
}

ImpDouble inner(const ImpDouble *p, const ImpDouble *q, const ImpInt k)
{
    return cblas_ddot(k, p, 1, q, 1);
}

void row_wise_inner(const Vec &V1, const Vec &V2, const ImpInt &row,
        const ImpInt &col,const ImpDouble &alpha, Vec &vv){
    const ImpDouble *v1p = V1.data(), *v2p = V2.data();

    #pragma omp parallel for schedule(guided)
    for(ImpInt i = 0; i < row; i++)
        vv[i] += alpha*inner(v1p+i*col, v2p+i*col, col);
}

void init_mat(Vec &vec, const ImpLong nr_rows, const ImpLong nr_cols) {
    default_random_engine ENGINE(rand());
    vec.resize(nr_rows*nr_cols, 0.1);
    uniform_real_distribution<ImpDouble> dist(-0.1*qrsqrt(nr_cols),0.1*qrsqrt(nr_cols));

    auto gen = std::bind(dist, ENGINE);
    generate(vec.begin(), vec.end(), gen);
}

void ImpData::read(bool has_label, const ImpLong *ds) {
    ifstream fs(file_name);
    string line, label_block, label_str;
    char dummy;

    ImpLong idx, y_nnz=0, x_nnz=0;
    ImpInt fid, label;
    ImpDouble val;

    while (getline(fs, line)) {
        m++;
        istringstream iss(line);

        if (has_label) {
            iss >> label_block;
            istringstream labelst(label_block);
            while (getline(labelst, label_str, ',')) {
                istringstream labels(label_str);
                labels >> idx >> dummy >> label;
                n = max(n, idx+1);
                y_nnz++;
            }
        }

        while (iss >> fid >> dummy >> idx >> dummy >> val) {
            f = max(f, fid+1);
            if (ds!= nullptr && ds[fid] <= idx)
                continue;
            x_nnz++;
        }
    }

    fs.clear();
    fs.seekg(0);

    nnz_x = x_nnz;
    N.resize(x_nnz);

    X.resize(m+1);
    Y.resize(m+1);

    if (has_label) {
        nnz_y = y_nnz;
        M.resize(y_nnz);
    }

    vector<ImpInt> nnx(m, 0), nny(m, 0);

    ImpLong nnz_i=0, nnz_j=0, i=0;

    while (getline(fs, line)) {
        istringstream iss(line);

        if (has_label) {
            iss >> label_block;
            istringstream labelst(label_block);
            while (getline(labelst, label_str, ',')) {
                nnz_j++;
                istringstream labels(label_str);
                labels >> idx >> dummy >> label;
                M[nnz_j-1].fid = (label > 0) ? 1 : -1;
                M[nnz_j-1].idx = idx;
            }
            nny[i] = nnz_j;
        }

        while (iss >> fid >> dummy >> idx >> dummy >> val) {
            if (ds!= nullptr && ds[fid] <= idx)
                continue;
            nnz_i++;
            N[nnz_i-1].fid = fid;
            N[nnz_i-1].idx = idx;
            N[nnz_i-1].val = val;
        }
        nnx[i] = nnz_i;
        i++;
    }

    X[0] = N.data();
    for (ImpLong i = 0; i < m; i++) {
        X[i+1] = N.data() + nnx[i];
    }

    if (has_label) {
        Y[0] = M.data();
        for (ImpLong i = 0; i < m; i++) {
            Y[i+1] = M.data() + nny[i];
        }
    }
    fs.close();
}

void ImpData::split_fields() {
    Ns.resize(f);
    Xs.resize(f);

    Ds.resize(f);
    freq.resize(f);

    vector<ImpLong> f_sum_nnz(f, 0);
    vector<vector<ImpLong>> f_nnz(f);

    for (ImpInt fi = 0; fi < f; fi++) {
        Ds[fi] = 0;
        f_nnz[fi].resize(m, 0);
        Xs[fi].resize(m+1);
    }

    for (ImpLong i = 0; i < m; i++) {
        for (Node* x = X[i]; x < X[i+1]; x++) {
            ImpInt fid = x->fid;
            f_sum_nnz[fid]++;
            f_nnz[fid][i]++;
        }
    }

    for (ImpInt fi = 0; fi < f; fi++) {
        Ns[fi].resize(f_sum_nnz[fi]);
        f_sum_nnz[fi] = 0;
    }

    for (ImpLong i = 0; i < m; i++) {
        for (Node* x = X[i]; x < X[i+1]; x++) {
            ImpInt fid = x->fid;
            ImpLong idx = x->idx;
            ImpDouble val = x->val;

            f_sum_nnz[fid]++;
            Ds[fid] = max(idx+1, Ds[fid]);
            ImpLong nnz_i = f_sum_nnz[fid]-1;

            Ns[fid][nnz_i].fid = fid;
            Ns[fid][nnz_i].idx = idx;
            Ns[fid][nnz_i].val = val;
        }
    }

    for(ImpInt fi = 0; fi < f; fi++){
        freq[fi].resize(Ds[fi]);
        fill(freq[fi].begin(), freq[fi].end(), 0);
    }

    for( ImpLong i = 0; i < m; i++){
        for(Node* x = X[i]; x < X[i+1]; x++){
            ImpInt fid = x->fid;
            ImpLong idx = x->idx;
            freq[fid][idx]++;
        }
    }
    for (ImpInt fi = 0; fi < f; fi++) {
        Node* fM = Ns[fi].data();
        Xs[fi][0] = fM;
        ImpLong start = 0;
        for (ImpLong i = 0; i < m; i++) {
            Xs[fi][i+1] = fM + start + f_nnz[fi][i];
            start += f_nnz[fi][i];
        }
    }

    X.clear();
    X.shrink_to_fit();

    N.clear();
    N.shrink_to_fit();
}

void ImpData::transY(const vector<YNode*> &YT) {
    n = YT.size() - 1;
    vector<pair<ImpLong, YNode*>> perm;
    ImpLong nnz = 0;
    vector<ImpLong> nnzs(m, 0);

    for (ImpLong i = 0; i < n; i++)
        for (YNode* y = YT[i]; y < YT[i+1]; y++) {
            if (y->idx >= m )
              continue;
            nnzs[y->idx]++;
            perm.emplace_back(i, y);
            nnz++;
        }

    auto sort_by_column = [&] (const pair<ImpLong, YNode*> &lhs,
            const pair<ImpLong, YNode*> &rhs) {
        return tie(lhs.second->idx, lhs.first) < tie(rhs.second->idx, rhs.first);
    };

    sort(perm.begin(), perm.end(), sort_by_column);

    M.resize(nnz);
    nnz_y = nnz;
    for (ImpLong nnz_i = 0; nnz_i < nnz; nnz_i++) {
        M[nnz_i].idx = perm[nnz_i].first;
        M[nnz_i].fid = perm[nnz_i].second->fid;
        M[nnz_i].val = perm[nnz_i].second->val;
    }

    Y[0] = M.data();
    ImpLong start_idx = 0;
    for (ImpLong i = 0; i < m; i++) {
        start_idx += nnzs[i];
        Y[i+1] = M.data()+start_idx;
    }
}

void ImpData::print_data_info() {
    cout << "File:";
    cout << file_name;
    cout.width(12);
    cout << "m:";
    cout << m;
    cout.width(12);
    cout << "n:";
    cout << n;
    cout.width(12);
    cout << "f:";
    cout << f;
    cout.width(12);
    cout << "d:";
    cout << Ds[0];
    cout << endl;
}

void ImpProblem::UTx(const Node* x0, const Node* x1, const Vec &A, ImpDouble *c) {
    for (const Node* x = x0; x < x1; x++) {
        const ImpLong idx = x->idx;
        const ImpDouble val = x->val;
        for (ImpInt d = 0; d < k; d++) {
            ImpLong jd = idx*k+d;
            c[d] += val*A[jd];
        }
    }
}

void ImpProblem::UTX(const vector<Node*> &X, const ImpLong m1, const Vec &A, Vec &C) {
    fill(C.begin(), C.end(), 0);
    ImpDouble* c = C.data();
#pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m1; i++)
        UTx(X[i], X[i+1], A, c+i*k);
}


void ImpProblem::init_pair(const ImpInt &f12,
        const ImpInt &fi, const ImpInt &fj,
        const shared_ptr<ImpData> &d1,
        const shared_ptr<ImpData> &d2) {
    const ImpLong Df1 = d1->Ds[fi];
    const ImpLong Df2 = d2->Ds[fj];

    const vector<Node*> &X1 = d1->Xs[fi];
    const vector<Node*> &X2 = d2->Xs[fj];

    init_mat(W[f12], Df1, k);
    init_mat(H[f12], Df2, k);
    P[f12].resize(d1->m*k, 0);
    Q[f12].resize(d2->m*k, 0);
    UTX(X1, d1->m, W[f12], P[f12]);
    UTX(X2, d2->m, H[f12], Q[f12]);
}

void ImpProblem::add_side(const Vec &p, const Vec &q, const ImpLong &m1, Vec &a1) {
    const ImpDouble *pp = p.data(), *qp = q.data();
    for (ImpLong i = 0; i < m1; i++) {
        const ImpDouble *pi = pp+i*k, *qi = qp+i*k;
        a1[i] += inner(pi, qi, k);
    }
}

void ImpProblem::calc_side() {
    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = f1; f2 < fu; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            add_side(P[f12], Q[f12], m, a);
        }
    }
    for (ImpInt f1 = fu; f1 < f; f1++) {
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            add_side(P[f12], Q[f12], n, b);
        }
    }
}

ImpDouble ImpProblem::calc_cross(const ImpLong &i, const ImpLong &j) {
    ImpDouble cross_value = 0.0;
    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = fu; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            const ImpDouble *pp = P[f12].data();
            const ImpDouble *qp = Q[f12].data();
            cross_value += inner(pp+i*k, qp+j*k, k);
        }
    }
    return cross_value;
}

void ImpProblem::init_y_tilde() {
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m; i++) {
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {
            ImpLong j = y->idx;
            y->val = a[i]+b[j]+calc_cross(i, j);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < n; j++) {
        for (YNode* y = V->Y[j]; y < V->Y[j+1]; y++) {
            ImpLong i = y->idx;
            y->val = a[i]+b[j]+calc_cross(i, j);
        }
    }

}

void ImpProblem::update_side(const bool &sub_type, const Vec &S
        , const Vec &Q1, Vec &W1, const vector<Node*> &X12, Vec &P1) {
    
    const ImpLong m1 = (sub_type)? m : n;
    // Update W1
    axpy( S.data(), W1.data(), S.size(), 1);

    // Update y_tilde and pq
    Vec &a1 = (sub_type)? a:b;
    shared_ptr<ImpData> U1 = (sub_type)? U:V;
    shared_ptr<ImpData> V1 = (sub_type)? V:U;

    Vec gaps(m1, 0);
    Vec XS(P1.size(), 0);
    UTX(X12, m1, S, XS);
    axpy( XS.data(), P1.data(), XS.size(), 1);
    row_wise_inner(XS, Q1, m1, k, 1, gaps);

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        a1[i] += gaps[i];
        for (YNode* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            y->val += gaps[i];
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < V1->m; j++) {
        for (YNode* y = V1->Y[j]; y < V1->Y[j+1]; y++) {
            const ImpLong i = y->idx;
            y->val += gaps[i];
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
}

void ImpProblem::update_cross(const bool &sub_type, const Vec &S,
        const Vec &Q1, Vec &W1, const vector<Node*> &X12, Vec &P1) {
    axpy( S.data(), W1.data(), S.size(), 1);
    const ImpLong m1 = (sub_type)? m : n;

    shared_ptr<ImpData> U1 = (sub_type)? U:V;
    shared_ptr<ImpData> V1 = (sub_type)? V:U;

    Vec XS(P1.size(), 0);
    UTX(X12, m1, S, XS);
    axpy( XS.data(), P1.data(), P1.size(), 1);

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        for (YNode* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            const ImpLong j = y->idx;
            y->val += inner( XS.data()+i*k, Q1.data()+j*k, k);
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < V1->m; j++) {
        for (YNode* y = V1->Y[j]; y < V1->Y[j+1]; y++) {
            const ImpLong i = y->idx;
            y->val += inner( XS.data()+i*k, Q1.data()+j*k, k);
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
}

void ImpProblem::init_y_imp(){
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U->m; i++) {
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {
            const ImpLong j = y->idx;
            for (ImpInt f1 = 0; f1 < f_imp; f1++) {
                for (ImpInt f2 = f1; f2 < f_imp; f2++) {
                    if( f1 >= fu_imp || f2 < fu_imp )
                        continue;
                    ImpLong f12 = index_vec(f1, f2, f_imp);
                    y->val_imp += inner(P_imp[f12].data() + i*k_imp, Q_imp[f12].data() + j*k_imp, k_imp);
                }
            }
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < V->m; j++) {
        for (YNode* y = V->Y[j]; y < V->Y[j+1]; y++) {
            const ImpLong i = y->idx;
            for (ImpInt f1 = 0; f1 < f_imp; f1++) {
                for (ImpInt f2 = f1; f2 < f_imp; f2++) {
                    if( f1 >= fu_imp || f2 < fu_imp )
                        continue;
                    ImpLong f12 = index_vec(f1, f2, f_imp);
                    y->val_imp += inner(P_imp[f12].data() + i*k_imp, Q_imp[f12].data() + j*k_imp, k_imp);
                }
            }
        }
    }
}

void ImpProblem::init() {
    lambda = param->lambda;
    w = param->omega;
    wn = param->omega_neg;

    m = U->m;
    n = V->m;

    fu = U->f;
    fv = V->f;
    f = fu+fv;

    k = param->k;

    a.resize(m, 0);
    b.resize(n, 0);

    sa.resize(m, 0);
    sb.resize(n, 0);

    const ImpInt nr_blocks = f*(f+1)/2;

    W.resize(nr_blocks);
    H.resize(nr_blocks);

    P.resize(nr_blocks);
    Q.resize(nr_blocks);

    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? U: V);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? U: V);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);
            const ImpInt f12 = index_vec(f1, f2, f);
            if( !param->self_side && ( f1 >= fu || f2 < fu ))
                continue;
            init_pair(f12, fi, fj, d1, d2);
        }
    }

    cache_sasb();

    if (param->self_side)
        calc_side();

    init_item_weights();
    init_y_tilde();

    init_L_pos();
    init_expyy();
    init_y_imp();
}

void ImpProblem::cache_sasb() {
    fill(sa.begin(), sa.end(), 0);
    fill(sb.begin(), sb.end(), 0);

    const Vec o1(m, 1), o2(n, 1);
    Vec tk(k);

    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = fu; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            const Vec &P1 = P[f12], &Q1 = Q[f12];

            fill(tk.begin(), tk.end(), 0);
            mv(Q1.data(), o2.data(), tk.data(), n, k, 0, true);
            mv(P1.data(), tk.data(), sa.data(), m, k, 1, false);

            fill(tk.begin(), tk.end(), 0);
            mv(P1.data(), o1.data(), tk.data(), m, k, 0, true);
            mv(Q1.data(), tk.data(), sb.data(), n, k, 1, false);
        }
    }
}

void ImpProblem::gd_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1,Vec &G) {
    fill(G.begin(), G.end(), 0);
    const shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    const ImpInt fi = (f1 < fu)? f1: f1-fu;
    if(param->freq){
        const ImpLong Df1 = U1->Ds[fi];
        vector<ImpLong> &freq = U1->freq[fi];
        assert( Df1 == freq.size());
        for(ImpLong i = 0; i < Df1; i++)
            axpy( W1.data()+i*k, G.data()+i*k, k, lambda * ImpDouble(freq[i]));
    }
    else{
        axpy( W1.data(), G.data(), G.size(), lambda);
    }
    gd_pos_cross(f1, Q1, W1, G);
    if (w != 0)
        gd_neg_cross(f1, Q1, W1, G);
}


void ImpProblem::hs_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V,
        const Vec &VQTQ, Vec &Hv, const Vec &Q1,
        const vector<Node*> &X, const vector<YNode*> &Y, Vec &Hv_) {
    hs_pos_cross(m1, n1, V, VQTQ, Hv, Q1, X, Y, Hv_);
    if (w != 0)
        hs_neg_cross(m1, n1, V, VQTQ, Hv, Q1, X, Y, Hv_);
}

void ImpProblem::cg(const ImpInt &f1, const ImpInt &f2, Vec &S1,
        const Vec &Q1, const Vec &G, Vec &P1) {

    const ImpInt base = (f1 < fu)? 0: fu;
    const ImpInt fi = f1-base;

    const shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    const vector<YNode*> &Y = U1->Y;
    const vector<Node*> &X = U1->Xs[fi];

    const ImpLong m1 = (f1 < fu)? m:n;
    const ImpLong n1 = (f1 < fu)? n:m;

    const ImpLong Df1 = U1->Ds[fi], Df1k = Df1*k;
    const ImpInt nr_threads = param->nr_threads;
    Vec Hv_(nr_threads*Df1k);

    ImpInt nr_cg = 0, max_cg = 20;
    ImpDouble g2 = 0, r2, cg_eps = 0.09, alpha = 0, beta = 0, gamma = 0, vHv;

    Vec V(Df1k, 0), R(Df1k, 0), Hv(Df1k, 0);
    Vec QTQ, VQTQ;

    if (!(f1 < fu && f2 < fu) && !(f1>=fu && f2>=fu) && w != 0) {
        QTQ.resize(k*k, 0);
        VQTQ.resize(Df1k, 0);
        mm(Q1.data(), Q1.data(), QTQ.data(), k, n1);
    }

    for (ImpLong jd = 0; jd < Df1k; jd++) {
        R[jd] = -G[jd];
        V[jd] = R[jd];
        g2 += G[jd]*G[jd];
    }

    r2 = g2;
    while (g2*cg_eps < r2 && nr_cg < max_cg) {
        nr_cg++;

        fill(Hv.begin(), Hv.end(), 0);
        fill(Hv_.begin(), Hv_.end(), 0);

        if(param->freq){
            vector<ImpLong> &freq = U1->freq[fi];
            assert( Df1 == freq.size());
            for(ImpLong i = 0; i < Df1; i++)
                axpy( V.data()+i*k, Hv.data()+i*k, k, lambda * ImpDouble(freq[i]));
        }
        else{
            axpy( V.data(), Hv.data(), V.size(), lambda);
        }

        if ((f1 < fu && f2 < fu) || (f1>=fu && f2>=fu)){
             cerr << "wrong" << endl;
        }
        else {
            if (w != 0)
                mm(V.data(), QTQ.data(), VQTQ.data(), Df1, k, k);
            hs_cross(m1, n1, V, VQTQ, Hv, Q1, X, Y, Hv_);
        }

        vHv = inner(V.data(), Hv.data(), Df1k);
        gamma = r2;
        alpha = gamma/vHv;
        axpy(V.data(), S1.data(), Df1k, alpha);
        axpy(Hv.data(), R.data(), Df1k, -alpha);
        r2 = inner(R.data(), R.data(), Df1k);
        beta = r2/gamma;
        scal(V.data(), Df1k, beta);
        axpy(R.data(), V.data(), Df1k, 1);
    }
}

void ImpProblem::solve_cross(const ImpInt &f1, const ImpInt &f2) {
    const ImpInt f12 = index_vec(f1, f2, f);
    const vector<Node*> &U1 = U->Xs[f1], &V1 = V->Xs[f2-fu];
    Vec &W1 = W[f12], &H1 = H[f12], &P1 = P[f12], &Q1 = Q[f12];

    Vec GW(W1.size(),0), GH(H1.size(),0);
    Vec SW(W1.size(),0), SH(H1.size(),0);

    gd_cross(f1, Q1, W1, GW);
    cg(f1, f2, SW, Q1, GW, P1);
    line_search(f1, f2, SW, Q1, W1, P1, GW);
    update_cross(true, SW, Q1, W1, U1, P1);

    gd_cross(f2, P1, H1, GH);
    cg(f2, f1, SH, P1, GH, Q1);
    line_search(f2, f1, SH, P1, H1, Q1, GH);
    update_cross(false, SH, P1, H1, V1, Q1);
}

void ImpProblem::one_epoch() {

    for (ImpInt f1 = 0; f1 < fu; f1++)
        for (ImpInt f2 = fu; f2 < f; f2++)
            solve_cross(f1, f2);

    if (param->self_side && w != 0)
        cache_sasb();
}

void ImpProblem::init_va(ImpInt size) {

    if (Uva->file_name.empty())
        return;

    mt = Uva->m;

    const ImpInt nr_blocks = f*(f+1)/2;

    Pva.resize(nr_blocks);
    Qva.resize(nr_blocks);

    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? Uva: V);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? Uva: V);
            const ImpInt f12 = index_vec(f1, f2, f);
            if( !param->self_side && ( f1 >= fu || f2 < fu ))
                continue;
            Pva[f12].resize(d1->m*k);
            Qva[f12].resize(d2->m*k);
        }
    }

    va_loss_prec.resize(size,0);
    va_loss_ndcg.resize(size,0);
    top_k.resize(size);
    ImpInt start = 5;

    cout << "iter";
    for (ImpInt i = 0; i < size; i++) {
        top_k[i] = start;
        cout.width(9);
        cout << "( p@ " << start << ", ";
        cout.width(6);
        cout << "nDCG@" << start << " )";
        start *= 2;
    }
    cout.width(12);
    cout << "logloss";
    cout << endl;
}

void ImpProblem::pred_z(const ImpLong i, ImpDouble *z) {
    for(ImpInt f1 = 0; f1 < fu; f1++) {
        for(ImpInt f2 = fu; f2 < f; f2++) {
            ImpInt f12 = index_vec(f1, f2, f);
            ImpDouble *p1 = Pva[f12].data()+i*k, *q1 = Qva[f12].data();
            mv(q1, p1, z, n, k, 1, false);
        }
    }
}

ImpDouble ImpProblem::pred_i_j(const ImpLong i, const ImpLong j) {
    ImpDouble res = 0;
    for(ImpInt f1 = 0; f1 < fu; f1++) {
        for(ImpInt f2 = fu; f2 < f; f2++) {
            ImpInt f12 = index_vec(f1, f2, f);
            ImpDouble *p1 = Pva[f12].data()+i*k, *q1 = Qva[f12].data()+j*k;
            res += inner(p1, q1, k);
        }
    }
    return res;
}

void ImpProblem::init_Pva_Qva_at_bt(){
    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? Uva: V);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);

        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? Uva: V);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);

            const ImpInt f12 = index_vec(f1, f2, f);
            if( !param->self_side && ( f1 >= fu || f2 < fu ))
                continue;
            UTX(d1->Xs[fi], d1->m, W[f12], Pva[f12]);
            UTX(d2->Xs[fj], d2->m, H[f12], Qva[f12]);
        }
    }

    at.resize(Uva->m, 0);
    bt.resize(V->m, 0);
    fill(at.begin(), at.end(), 0);
    fill(bt.begin(), bt.end(), 0);

    if (param->self_side) {
        for (ImpInt f1 = 0; f1 < fu; f1++) {
            for (ImpInt f2 = f1; f2 < fu; f2++) {
                const ImpInt f12 = index_vec(f1, f2, f);
                add_side(Pva[f12], Qva[f12], Uva->m, at);
            }
        }
        for (ImpInt f1 = fu; f1 < f; f1++) {
            for (ImpInt f2 = f1; f2 < f; f2++) {
                const ImpInt f12 = index_vec(f1, f2, f);
                add_side(Pva[f12], Qva[f12], V->m, bt);
            }
        }
    }
}

void ImpProblem::save_Pva_Qva(string &model_path){
    init_Pva_Qva_at_bt();
    ofstream of(model_path, ios::out | ios::trunc | ios::binary);
    of.write( reinterpret_cast<char*>(&fu), sizeof(ImpInt) );
    of.write( reinterpret_cast<char*>(&fv), sizeof(ImpInt) );
    of.write( reinterpret_cast<char*>(&f), sizeof(ImpInt) );
    of.write( reinterpret_cast<char*>(&k), sizeof(ImpInt) );
    for (ImpInt f1 = 0; f1 < f; f1++) {
        for (ImpInt f2 = f1; f2 < f; f2++) {
            ImpInt f12 = index_vec(f1, f2, f);
            if( f1 >= fu || f2 < fu )
                continue;
            ImpLong Pva_size = Pva[f12].size();
            ImpLong Qva_size = Qva[f12].size();
            of.write( reinterpret_cast<char*>(&f12), sizeof(ImpInt) );
            of.write( reinterpret_cast<char*>(&Pva_size), sizeof(ImpLong) );
            of.write( reinterpret_cast<char*>(&Qva_size), sizeof(ImpLong) );

            of.write( reinterpret_cast<char*>(Pva[f12].data()), sizeof(ImpDouble)*Pva_size );
            of.write( reinterpret_cast<char*>(Qva[f12].data()), sizeof(ImpDouble)*Qva_size );
        }
    }
    of.close();
}

void ImpProblem::load_imputation_model(string &model_imp_path){
    ifstream ifile(model_imp_path, ios::in | ios::binary);
    ifile.read( reinterpret_cast<char*>(&fu_imp), sizeof(ImpInt) );
    ifile.read( reinterpret_cast<char*>(&fv_imp), sizeof(ImpInt) );
    ifile.read( reinterpret_cast<char*>(&f_imp), sizeof(ImpInt) );
    ifile.read( reinterpret_cast<char*>(&k_imp), sizeof(ImpInt) );
    P_imp.resize(f_imp*(f_imp+1)/2);
    Q_imp.resize(f_imp*(f_imp+1)/2);
    for (ImpInt f1 = 0; f1 < f_imp; f1++) {
        for (ImpInt f2 = f1; f2 < f_imp; f2++) {
            if( f1 >= fu_imp || f2 < fu_imp )
                continue;
            ImpInt f12;
            ImpLong P_size, Q_size;
            ifile.read( reinterpret_cast<char*>(&f12), sizeof(ImpInt) );
            ifile.read( reinterpret_cast<char*>(&P_size), sizeof(ImpLong) );
            ifile.read( reinterpret_cast<char*>(&Q_size), sizeof(ImpLong) );
            assert( f12 == index_vec(f1, f2, f_imp) );
            P_imp[f12].resize(P_size);
            Q_imp[f12].resize(Q_size);
            ifile.read( reinterpret_cast<char*>(P_imp[f12].data()), sizeof(ImpDouble)*P_size );
            ifile.read( reinterpret_cast<char*>(Q_imp[f12].data()), sizeof(ImpDouble)*Q_size );
        }
    }
    ifile.close();
}

void ImpProblem::calc_gauc(){
    ImpDouble gauc_sum = 0;
    ImpDouble gauc_weight_sum = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: gauc_sum, gauc_weight_sum)
    for (ImpLong i = 0; i < Uva->m; i++){
        ImpLong num_obv = ImpLong(Uva->Y[i+1] - Uva->Y[i]);
        Vec z(num_obv, 0);
        Vec label(num_obv, 0);
        ImpLong k = 0;
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++, k++){
            const ImpLong j = y->idx;
            z[k] += pred_i_j(i, j) + bt[j];
            label[k] = y->fid;
        }
        ImpDouble gauc_i = calc_auc_i(z,label);
        if(gauc_i != -1){
            gauc_sum += num_obv * gauc_i;
            gauc_weight_sum += num_obv;
        }
    }
    gauc = gauc_sum/gauc_weight_sum;
}

void ImpProblem::calc_auc(){
    vector<Vec> Zs(Uva->m);
    vector<Vec> labels(Uva->m);
    ImpLong sample_count = 0; 
    #pragma omp parallel for schedule(dynamic) reduction(+: sample_count)
    for (ImpLong i = 0; i < Uva->m; i++){
        ImpLong num_obv = ImpLong(Uva->Y[i+1] - Uva->Y[i]);
        Vec& z = Zs[i];
        Vec& label = labels[i];
        z.resize(num_obv);
        label.resize(num_obv);
        fill(z.begin(), z.end(), 0);
        fill(label.begin(), label.end(), 0);
        ImpLong k = 0;
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++, k++){
            const ImpLong j = y->idx;
            z[k] += pred_i_j(i, j) + bt[j];
            label[k] = y->fid;
            sample_count++;
        }
    }
    Vec Z_all, label_all;
    Z_all.reserve(sample_count);
    label_all.reserve(sample_count);
    for(ImpLong i = 0; i < Uva->m; i++){
        Z_all.insert(Z_all.end(), Zs[i].begin(), Zs[i].end());
        label_all.insert(label_all.end(), labels[i].begin(), labels[i].end());
    }
    auc = calc_auc_i(Z_all, label_all);
}

void ImpProblem::logloss() {
    ImpDouble tr_loss_t = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: tr_loss_t)
    for(ImpLong i = 0; i < m; i++){
        for(YNode *y = U->Y[i]; y < U->Y[i+1]; y++){

            const ImpDouble yy = y->val * (ImpDouble) y->fid;
            const ImpDouble w2 = (y->fid > 0)? 1 : wn;
            const ImpDouble iw = param->item_weight? item_w[y->idx]: 1;

            if( -yy > 0 )
                tr_loss_t += iw*w2 *(-yy + log1p( exp(yy) ));
            else
                tr_loss_t += iw*w2 * log1p( exp(-yy) );
        }
    }
    tr_loss = param->item_weight? tr_loss_t/n : tr_loss_t/U->M.size();

    ImpDouble ploss = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: ploss)
    for (ImpLong i = 0; i < Uva->m; i++) {
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            const ImpLong j = y->idx;
            ImpDouble w2 = (y->fid > 0)? 1 : wn;
            ImpDouble yy = (pred_i_j(i, j) + at[i])*y->fid;
            if (-yy > 0)
                ploss += w2 *(-yy + log1p( exp(yy) ));
            else
                ploss += w2 * log1p( exp(-yy) );
        }
    }
    loss = ploss/Uva->M.size();
}

void ImpProblem::validate() {
    const ImpInt nr_th = param->nr_threads, nr_k = top_k.size();
    ImpLong valid_samples = 0;

    vector<ImpLong> hit_counts(nr_th*nr_k, 0);
    vector<ImpDouble> ndcg_scores(nr_th*nr_k, 0);

    Vec va_item_w(n, 0);
    for (ImpLong i = 0; i < Uva->m; i++) {
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            const ImpLong j = y->idx;
            va_item_w[j]++;
        }
    }
    for (ImpLong j = 0; j < n; j++) {
        va_item_w[j] = (va_item_w[j] > 0)? 1/va_item_w[j]: 1;
    }

    ImpDouble tr_loss_t = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: tr_loss_t)
    for(ImpLong i = 0; i < m; i++){
        for(YNode *y = U->Y[i]; y < U->Y[i+1]; y++){

            const ImpDouble yy = y->val * (ImpDouble) y->fid;
            const ImpDouble w2 = (y->fid > 0)? 1 : wn;
            const ImpDouble iw = param->item_weight? item_w[y->idx]: 1;

            if( -yy > 0 )
                tr_loss_t += iw*w2 *(-yy + log1p( exp(yy) ));
            else
                tr_loss_t += iw*w2 * log1p( exp(-yy) );
        }
    }

    tr_loss = param->item_weight? tr_loss_t/n : tr_loss_t/U->M.size();

    ImpDouble ploss = 0;
    ImpDouble gauc_sum = 0, gauc_all_sum = 0;
    ImpDouble gauc_all_weight_sum = 0, gauc_weight_sum = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: valid_samples, ploss, gauc_all_sum, gauc_sum, gauc_all_weight_sum, gauc_weight_sum)
    for (ImpLong i = 0; i < Uva->m; i++) {
        Vec z_copy, z(bt);

        pred_z(i, z.data());

        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            const ImpLong j = y->idx;
            const ImpDouble yy = (z[j]+at[i])*y->fid;

            const ImpDouble w2 = (y->fid > 0)? 1 : wn;

            if (-yy > 0)
                ploss += w2 *(-yy + log1p( exp(yy) ));
            else
                ploss += w2 * log1p( exp(-yy) );
        }
        
        ImpDouble gauc_i = calc_gauc_i(z, i, true);
        if( gauc_i != -1 ){
            gauc_all_sum += gauc_i;
            gauc_all_weight_sum += 1;
        }
        gauc_i = calc_gauc_i(z, i, false);
        if( gauc_i != -1){
            gauc_sum += gauc_i * (ImpDouble)(Uva->Y[i+1] - Uva->Y[i]);
            gauc_weight_sum += (Uva->Y[i+1] - Uva->Y[i]);
        }

        z_copy.assign(z.begin(), z.end());
        prec_k(z.data(), i, top_k, hit_counts);
        ndcg(z_copy.data(), i, ndcg_scores);
        valid_samples++;
    }

    gauc_all = gauc_all_sum / gauc_all_weight_sum;
    gauc = gauc_sum / gauc_weight_sum;
    loss = ploss/Uva->M.size();

    fill(va_loss_prec.begin(), va_loss_prec.end(), 0);
    fill(va_loss_ndcg.begin(), va_loss_ndcg.end(), 0);
    for (ImpInt i = 0; i < nr_k; i++) {

        for (ImpLong num_th = 0; num_th < nr_th; num_th++){
            va_loss_prec[i] += hit_counts[i+num_th*nr_k];
            va_loss_ndcg[i] += ndcg_scores[i+num_th*nr_k];
        }

        va_loss_prec[i] /= ImpDouble(valid_samples*top_k[i]);
        va_loss_ndcg[i] /= ImpDouble(valid_samples);
    }
}

void ImpProblem::ndcg(ImpDouble *z, ImpLong i, vector<ImpDouble> &ndcg_scores) {
    ImpInt valid_count = 0;
    const ImpInt nr_k = top_k.size();
    vector<ImpDouble> dcg_score(nr_k, 0);
    vector<ImpDouble> idcg_score(nr_k, 0);

    ImpInt num_th = omp_get_thread_num();

    ImpLong max_z_idx = min(U->n, n);
    ImpLong num_of_pos = 0;
    for (YNode* nd = Uva->Y[i]; nd < Uva->Y[i+1]; nd++) {
        if( nd->fid == 1)
            num_of_pos++;
    }
    for (ImpInt state = 0; state < nr_k; state++) {
        while(valid_count < top_k[state]) {
            if ( valid_count >= max_z_idx )
               break;
            ImpLong argmax = distance(z, max_element(z, z + max_z_idx));
            z[argmax] = MIN_Z;
            for (YNode* nd = Uva->Y[i]; nd < Uva->Y[i+1]; nd++) {
                if (argmax == nd->idx && nd->fid == 1) {
                    dcg_score[state] += 1.0 / log2(valid_count + 2);
                    break;
                }
            }

            if( num_of_pos > valid_count )
                idcg_score[state] += 1.0 / log2(valid_count + 2);
            valid_count++;
        }
    }
    for (ImpInt i = 1; i < nr_k; i++) {
        dcg_score[i] += dcg_score[i-1];
        idcg_score[i] += idcg_score[i-1];
    }

    for (ImpInt i = 0; i < nr_k; i++) {
        ndcg_scores[i+num_th*nr_k] += dcg_score[i] / idcg_score[i];
    }
}

class Comp{
    const ImpDouble *dec_val;
    public:
    Comp(const ImpDouble *ptr): dec_val(ptr){}
    bool operator()(int i, int j) const{
        return dec_val[i] < dec_val[j];
    }
};

ImpDouble ImpProblem::calc_gauc_i(Vec &z, ImpLong i, bool do_sum_all){
    ImpDouble rank_sum  = 0;
    ImpDouble auc  = 0;
    ImpLong size = z.size();
    vector<ImpLong> indices(size);

    for(ImpLong j = 0; j < size; j++) indices[j] = j;

    sort(indices.begin(), indices.end(), Comp(z.data()));

    ImpLong tp = 0,fp = 0;
    ImpLong rank = 0;
    for(ImpLong j = 0; j < size; j++) {
        bool is_pos = false;
        bool is_obs = false;
        ImpLong idx = indices[j];
        for(YNode *y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            if(y->idx == idx)
                is_obs = true;
            if(y->idx == idx && y->fid > 0){
                is_pos = true; 
                break;
            }
        }

        if( !do_sum_all  && !is_obs)
            continue;

        if(is_pos){ 
            tp++;
            rank_sum += (rank + 1);
        }
        else{
            fp++;
        }

        rank += 1;
    }

    if(tp == 0 || fp == 0)
    {
        auc = -1;
    }
    else
        auc = (rank_sum - ((ImpDouble)tp + 1.0) * (ImpDouble)tp / 2.0)/ (ImpDouble)tp / (ImpDouble)fp;

    return auc;
}

ImpDouble ImpProblem::calc_auc_i(Vec &z, Vec &label){
    ImpDouble rank_sum  = 0;
    ImpDouble auc  = 0;
    ImpLong size = z.size();
    vector<ImpLong> indices(size);

    for(ImpLong j = 0; j < size; j++) indices[j] = j;
    sort(indices.begin(), indices.end(), Comp(z.data()));

    ImpLong tp = 0,fp = 0;
    ImpLong rank = 0;
    for(ImpLong j = 0; j < size; j++) {
        bool is_pos = (label[indices[j]] > 0)? true : false;

        if(is_pos){ 
            tp++;
            rank_sum += (rank + 1);
        }
        else{
            fp++;
        }

        rank += 1;
    }

    if(tp == 0 || fp == 0)
    {
        auc = -1;
    }
    else
        auc = (rank_sum - ((ImpDouble)tp + 1.0) * (ImpDouble)tp / 2.0)/ (ImpDouble)tp / (ImpDouble)fp;

    return auc;
}

void ImpProblem::prec_k(ImpDouble *z, ImpLong i, vector<ImpInt> &top_k, vector<ImpLong> &hit_counts) {
    ImpInt valid_count = 0;
    const ImpInt nr_k = top_k.size();
    vector<ImpLong> hit_count(nr_k, 0);

    ImpInt num_th = omp_get_thread_num();

    for (ImpInt state = 0; state < nr_k; state++) {
        while(valid_count < top_k[state]) {
            ImpLong argmax = distance(z, max_element(z, z+n));
            z[argmax] = MIN_Z;
            for (YNode* nd = Uva->Y[i]; nd < Uva->Y[i+1]; nd++) {
                if (! (nd->fid > 0))
                    continue;
                if (argmax == nd->idx) {
                    hit_count[state]++;
                    break;
                }
            }
            valid_count++;
        }
    }

    for (ImpInt i = 1; i < nr_k; i++) {
        hit_count[i] += hit_count[i-1];
    }
    for (ImpInt i = 0; i < nr_k; i++) {
        hit_counts[i+num_th*nr_k] += hit_count[i];
    }
}

void ImpProblem::print_epoch_info(ImpInt t) {
    ImpInt nr_k = top_k.size();
    cout.width(2);
    cout << t+1 << " ";
    if (!Uva->file_name.empty() && (t+1) % 2 == 0){
        init_Pva_Qva_at_bt();
        logloss();
        calc_auc();
        for (ImpInt i = 0; i < nr_k; i++ ) {
            cout.width(9);
            cout << "( " <<setprecision(3) << va_loss_prec[i]*100 << " ,";
            cout.width(6);
            cout << setprecision(3) << va_loss_ndcg[i]*100 << " )";
        }
        cout.width(13);
        cout << setprecision(3) << loss;
        cout << endl;
        cout << "tr_loss: " << tr_loss <<  " gauc: " << gauc << " gauc_all: " << gauc_all << " auc: " << auc << endl;
    }
}

void ImpProblem::solve() {
    init_va(5);
    for (ImpInt iter = 0; iter < param->nr_pass; iter++) {
        one_epoch();
        print_epoch_info(iter);
    }
}

void ImpProblem::write_header(ofstream &f_out) const{
    f_out << f << endl;
    f_out << fu << endl;
    f_out << fv << endl;
    f_out << k << endl;
    
    for(ImpInt fi = 0; fi < fu ; fi++)
        f_out << U->Ds[fi] << endl;
    
    for(ImpInt fi = 0; fi < fv ; fi++)
        f_out << V->Ds[fi] << endl;
}

void write_block(const Vec& block, const ImpLong& num_of_rows, const ImpInt& num_of_columns, char block_type, const ImpInt fi, const ImpInt fj, ofstream &f_out){
    ostringstream stringStream;
    stringStream << block_type << ',' << fi << ',' << fj;
    string line_info =  stringStream.str();

    for( ImpLong row_i = 0; row_i < num_of_rows; row_i++ ){
        f_out << line_info << ',' << row_i;
        ImpLong offset = row_i * num_of_columns;
        for(ImpInt col_i = 0; col_i < num_of_columns ; col_i++ ){
            f_out << " " <<block[offset + col_i];
        }
        f_out << endl;
    }
}

void ImpProblem::write_W_and_H(ofstream &f_out) const{
    for(ImpInt fi = 0; fi < f ; fi++){
        for(ImpInt fj = fi; fj < f; fj++){
            ImpInt fij = index_vec(fi, fj, f);
            ImpInt fi_base = (fi >= fu )? fi - fu : fi;
            ImpInt fj_base = (fj >= fu )? fj - fu : fj;
            if ( fi < fu && fj < fu ){
                if( !param->self_side )
                    continue;
                write_block(W[fij], U->Ds[fi_base], k, 'W', fi, fj, f_out);
                write_block(H[fij], U->Ds[fj_base], k, 'H', fi, fj, f_out);
            }
            else if (fi < fu && fj >= fu){
                write_block(W[fij], U->Ds[fi_base], k, 'W', fi, fj, f_out);
                write_block(H[fij], V->Ds[fj_base], k, 'H', fi, fj, f_out);
            }
            else if( fi >= fu && fj >= fu){
                if( !param->self_side )
                    continue;
                write_block(W[fij], V->Ds[fi_base], k, 'W', fi, fj, f_out);
                write_block(H[fij], V->Ds[fj_base], k, 'H', fi, fj, f_out);
            }
        }
    }

}

void ImpProblem::save_model(string & model_path ){
    ofstream f_out(model_path, ios::out | ios::trunc );
    write_header( f_out );
    write_W_and_H( f_out );
}

ImpDouble ImpProblem::pq(const ImpInt &i, const ImpInt &j,const ImpInt &f1, const ImpInt &f2) {
    ImpInt f12 = index_vec(f1, f2, f);
    ImpInt Pi = (f1 < fu)? i : j;
    ImpInt Qj = (f2 < fu)? i : j;
    ImpDouble  *pp = P[f12].data()+Pi*k, *qp = Q[f12].data()+Qj*k;
    return inner(qp, pp, k);
}

ImpDouble ImpProblem::norm_block(const ImpInt &f1,const ImpInt &f2) {
    ImpInt f12 = index_vec(f1, f2, f);
    Vec &W1 = W[f12], H1 = H[f12];
    ImpDouble res = 0;
    res += inner(W1.data(), W1.data(), W1.size());
    res += inner(H1.data(), H1.data(), H1.size());
    return res;
}

ImpDouble ImpProblem::l_pos_grad(const YNode *y, const ImpDouble iw) {
    const ImpDouble w2 = (y->fid > 0)? 1 : wn;
    const ImpDouble y_ij = y->fid, y_hat = y->val, expyy = y->expyy, y_imp = y->val_imp;
    return  iw*w2 * -y_ij / (1 + expyy) - w * (y_hat - y_imp);
}

ImpDouble ImpProblem::l_pos_hessian(const YNode *y, const ImpDouble iw) {
    const ImpDouble w2 = (y->fid > 0)? 1 : wn;
    const ImpDouble expyy = y->expyy;
    return iw*w2 * expyy / (1 + expyy) / (1 + expyy) - w;
}

void ImpProblem::init_expyy() {
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m; i++) {
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < n; j++) {
        for (YNode* y = V->Y[j]; y < V->Y[j+1]; y++) {
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
}

void ImpProblem::gd_pos_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, Vec &G) {
    const ImpLong &m1 = (f1 < fu)? m:n;

    const shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    const ImpInt fi = (f1 < fu)? f1 : f1 - fu;
    const vector<Node*> &X = U1->Xs[fi];
    const vector<YNode*> &Y = U1->Y;

    const ImpLong block_size = G.size();
    const ImpInt nr_threads = param->nr_threads;
    Vec G_(nr_threads*block_size, 0);
    
    const ImpDouble *qp = Q1.data();

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m1; i++) {
        Vec pk(k, 0);
        const ImpInt id = omp_get_thread_num();
        for (YNode* y = Y[i]; y < Y[i+1]; y++) {
            const ImpLong idx = (f1 < fu)? y->idx: i;
            const ImpLong j = y->idx;
            const ImpDouble *q1 = qp+j*k;

            const ImpDouble iw = param->item_weight? item_w[idx]: 1;
            const ImpDouble scale = l_pos_grad(y, iw);

            for (ImpInt d = 0; d < k; d++)
                pk[d] += scale*q1[d];
        }

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++) {
                const ImpLong jd = idx*k+d;
                G_[jd+id*block_size] += pk[d]*val;
            }
        }
    }
    for(ImpInt i = 0; i < nr_threads; i++)
        axpy(G_.data()+i*block_size, G.data(), block_size, 1);
}

void ImpProblem::gd_neg_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, Vec &G) {
    const Vec &a1 = (f1 < fu)? a: b;
    const Vec &b1 = (f1 < fu)? b: a;

    const vector<Vec> &Ps = (f1 < fu)? P:Q;
    const vector<Vec> &Qs = (f1 < fu)? Q:P;
    const vector<Vec> &Ps_imp = (f1 < fu)? P_imp:Q_imp;
    const vector<Vec> &Qs_imp = (f1 < fu)? Q_imp:P_imp;

    const ImpLong &m1 = (f1 < fu)? m:n;
    const ImpLong &n1 = (f1 < fu)? n:m;

    const shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    const ImpInt fi = (f1 < fu)? f1 : f1 - fu;
    const vector<Node*> &X = U1->Xs[fi];

    Vec QTQ(k*k, 0), T(m1*k, 0), o1(n1, 1), oQ(k, 0), bQ(k, 0);

    mv(Q1.data(), o1.data(), oQ.data(), n1, k, 0, true);
    mv(Q1.data(), b1.data(), bQ.data(), n1, k, 0, true);

    for (ImpInt al = 0; al < fu; al++) {
        for (ImpInt be = fu; be < f; be++) {
            const ImpInt fab = index_vec(al, be, f);
            const Vec &Qa = Qs[fab], &Pa = Ps[fab];
            mm(Qa.data(), Q1.data(), QTQ.data(), k, n1);
            mm(Pa.data(), QTQ.data(), T.data(), m1, k, k, 1);
        }
    }
    
    Vec QTQ_imp(k_imp*k, 0), T_imp(m1*k, 0);
    for (ImpInt al = 0; al < fu_imp; al++) {
        for (ImpInt be = fu_imp; be < f_imp; be++) {
            const ImpInt fab = index_vec(al, be, f_imp);
            const Vec &Qa_imp = Qs_imp[fab], &Pa_imp = Ps_imp[fab];
            mTm(k_imp, k, n1, Qa_imp.data(), Q1.data(), QTQ_imp.data(), k_imp, k, k, 1);
            mm(m1, k, k_imp, Pa_imp.data(), QTQ_imp.data(), T_imp.data(), k_imp, k, k, 1);
        }
    }

    const ImpLong block_size = G.size();
    const ImpInt nr_threads = param->nr_threads;
    Vec G_(nr_threads*block_size, 0);

    const ImpDouble *tp = T.data();
    const ImpDouble *tp_imp = T_imp.data();

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m1; i++) {
        const ImpInt id = omp_get_thread_num();
        const ImpDouble *t1 = tp+i*k;
        const ImpDouble *t1_imp = tp_imp+i*k;
        const ImpDouble z_i = a1[i];
        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++) {
                const ImpLong jd = idx*k+d;
                G_[jd+id*block_size] += w*(t1[d]-t1_imp[d]+z_i*oQ[d]+bQ[d])*val;
            }
        }
    }

    Gneg.resize(G.size());
    fill(Gneg.begin(), Gneg.end(), 0);
    for(ImpInt i = 0; i < nr_threads; i++){
        axpy(G_.data()+i*block_size, G.data(), block_size, 1);
        axpy(G_.data()+i*block_size, Gneg.data(), block_size, 1);
    }
}

void ImpProblem::hs_pos_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V,
        const Vec &VQTQ, Vec &Hv, const Vec &Q1,
        const vector<Node*> &X, const vector<YNode*> &Y, Vec &Hv_) {
    
    fill(Hv_.begin(), Hv_.end(), 0);
    const ImpDouble *qp = Q1.data();

    const ImpLong block_size = Hv.size();
    const ImpInt nr_threads = param->nr_threads;

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m1; i++) {
        const ImpInt id = omp_get_thread_num();

        Vec phi(k, 0), ka(k, 0);
        UTx(X[i], X[i+1], V, phi.data());

        for (YNode* y = Y[i]; y < Y[i+1]; y++) {

            const ImpLong j = y->idx;
            const ImpDouble *dp = qp + j*k;

            const ImpLong idx = (m1 == m)? j: i;
            const ImpDouble iw = param->item_weight? item_w[idx]: 1;

            const ImpDouble val = inner(phi.data(), dp, k) * l_pos_hessian(y, iw);

            for (ImpInt d = 0; d < k; d++)
                ka[d] += val*dp[d];
        }

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++) {
                const ImpLong jd = idx*k+d;
                Hv_[jd+id*block_size] += ka[d]*val;
            }
        }
    }

    for(ImpInt i = 0; i < nr_threads; i++)
        axpy(Hv_.data()+i*block_size, Hv.data(), block_size, 1);
}

void ImpProblem::hs_neg_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V,
        const Vec &VQTQ, Vec &Hv, const Vec &Q1,
        const vector<Node*> &X, const vector<YNode*> &Y, Vec &Hv_) {

    fill(Hv_.begin(), Hv_.end(), 0);
    const ImpLong block_size = Hv.size();
    const ImpInt nr_threads = param->nr_threads;

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m1; i++) {
        const ImpInt id = omp_get_thread_num();
        Vec tau(k, 0);
        UTx(X[i], X[i+1], VQTQ, tau.data());

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt d = 0; d < k; d++) {
                const ImpLong jd = idx*k+d;
                Hv_[jd+id*block_size] += w*tau[d]*val;
            }
        }
    }

    for(ImpInt i = 0; i < nr_threads; i++)
        axpy(Hv_.data()+i*block_size, Hv.data(), block_size, 1);
}

void ImpProblem::line_search(const ImpInt &f1, const ImpInt &f2, Vec &S1,
        const Vec &Q1, const Vec &W1, Vec &P1, const Vec &G) {
    const ImpInt base = (f1 < fu)? 0: fu;
    const ImpInt fi = f1-base;

    shared_ptr<ImpData> U1 = (f1 < fu)? U:V;
    vector<YNode*> &Y = U1->Y;
    const vector<Node*> &X = U1->Xs[fi];

    const ImpLong m1 = (f1 < fu)? m:n;
    const ImpLong n1 = (f1 < fu)? n:m;

    const ImpLong Df1 = U1->Ds[fi], Df1k = Df1*k;
    const ImpInt nr_threads = param->nr_threads;
    Vec Hs_(nr_threads*Df1k);

    ImpDouble sTg_neg=0, sHs=0, sTg, wTs=0, sTs=0;

    sTg = inner(S1.data(), G.data(), S1.size());
    if(param->freq){
        vector<ImpLong> &freq = U1->freq[fi];
        assert( Df1 == freq.size());
        for(ImpLong i = 0; i < Df1; i++) {
            wTs += inner( S1.data()+i*k, W1.data()+i*k, k)*lambda * ImpDouble(freq[i]);
            sTs += inner( S1.data()+i*k, S1.data()+i*k, k)*lambda * ImpDouble(freq[i]);
        }
    }
    else{
        wTs = lambda*inner(S1.data(), W1.data(), S1.size());
        sTs = lambda*inner(S1.data(), S1.data(), S1.size());
    }

    if (w != 0) {
        sTg_neg = inner(S1.data(), Gneg.data(), S1.size());

        Vec Hs(Df1k, 0);
        Vec QTQ, SQTQ;
        if (!(f1 < fu && f2 < fu) && !(f1>=fu && f2>=fu)) {
            QTQ.resize(k*k, 0);
            SQTQ.resize(Df1k, 0);
            mm(Q1.data(), Q1.data(), QTQ.data(), k, n1);
        }
        fill(Hs.begin(), Hs.end(), 0);
        fill(Hs_.begin(), Hs_.end(), 0);
        if ((f1 < fu && f2 < fu) || (f1>=fu && f2>=fu)){
            cerr << "wrong" << endl;
        }
        else {
            mm(S1.data(), QTQ.data(), SQTQ.data(), Df1, k, k);
            hs_neg_cross(m1, n1, S1, SQTQ, Hs, Q1, X, Y, Hs_);
        }
        sHs = inner(S1.data(), Hs.data(), S1.size());
    }

    Vec XS(P1.size(), 0);
    UTX(X, m1, S1, XS);
    if ( (f1 < fu && f2 < fu) || (f1 >=fu && f2 >= fu))
        calc_delta_y_side(Y, m1, XS, Q1);
    else
        calc_delta_y_cross(Y, m1, XS, Q1);
    ImpDouble theta = 1, beta = 0.5, nu = 0.1;
    while(true){
        if(theta < 1e-20){
            scal(S1.data(), S1.size(), 0);
            cerr << "Step size too small and skip this block." << endl;
            break;
        }
        ImpDouble L_pos_new = calc_L_pos(Y, m1, theta);
        ImpDouble delta = L_pos_new - L_pos + theta * sTg_neg + 0.5 * theta * theta * sHs + (theta*wTs + 0.5*theta*theta*sTs);
        if( delta <= nu * theta * sTg ){
            L_pos = L_pos_new;
            scal(S1.data(), S1.size(), theta);
            break;
        }
        theta *= beta;
    }
    if( theta != 1 )
        cerr << "Do line search " << theta << endl << flush;
}

void ImpProblem::calc_delta_y_side(vector<YNode*> &Y, const ImpLong m1, const Vec &XS, const Vec &Q){
    const ImpDouble *qp = Q.data();
    const ImpDouble *pp = XS.data();
    #pragma omp parallel for schedule(dynamic)
    for(ImpLong i = 0 ; i < m1; i++){
        ImpDouble delta =  inner(pp + i * k, qp + i * k, k);
        for(YNode *y = Y[i]; y != Y[i+1]; y++){
            y->delta = delta;
        }
    }
}
void ImpProblem::calc_delta_y_cross(vector<YNode*> &Y, const ImpLong m1, const Vec &XS, const Vec &Q){
    const ImpDouble *qp = Q.data();
    const ImpDouble *pp = XS.data();
    #pragma omp parallel for schedule(dynamic)
    for(ImpLong i = 0 ; i < m1; i++){
        for(YNode *y = Y[i]; y != Y[i+1]; y++){
            ImpLong j = y->idx;
            y->delta = inner(pp + i * k, qp + j * k, k);
        }
    }
}

ImpDouble ImpProblem::calc_L_pos(vector<YNode*> &Y, const ImpLong m1, const ImpDouble theta){
    ImpDouble L_pos_new = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: L_pos_new)
    for(ImpLong i = 0; i < m1; i++){
        for(YNode *y = Y[i]; y != Y[i+1]; y++){
            const ImpDouble y_hat_new = y->val + theta * y->delta;
            const ImpDouble y_imp = y->val_imp;
            const ImpDouble yy = y_hat_new * (ImpDouble) y->fid;
            const ImpDouble w2 = (y->fid > 0)? 1 : wn;

            const ImpLong idx = (m1 == m)? y->idx: i;
            const ImpDouble iw = param->item_weight? item_w[idx]: 1;

            if( -yy > 0 )
                L_pos_new += iw*w2 * (-yy + log1p( exp(yy) )) - 0.5 * w * (y_hat_new - y_imp) * (y_hat_new - y_imp);
            else
                L_pos_new += iw*w2 * log1p( exp(-yy) ) - 0.5 * w * (y_hat_new - y_imp) * (y_hat_new - y_imp);
        }
    }
    return L_pos_new;
}
    
void ImpProblem::init_L_pos(){
    ImpDouble res = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: res)
    for (ImpLong i = 0; i < m; i++) {
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {

            const ImpDouble iw = param->item_weight? item_w[y->idx]: 1;
            const ImpDouble w2 = (y->fid > 0)? 1 : wn;
            const ImpDouble yy = y->val * (ImpDouble) y->fid;

            if( -yy > 0 )
                res += iw * w2 * (-yy + log1p( exp(yy) )) - 0.5 * w * (y->val - y->val_imp) * (y->val - y->val_imp);
            else
                res += iw * w2 * log1p( exp(-yy) ) - 0.5 * w * (y->val - y->val_imp) * (y->val - y->val_imp);
        }
    }
    L_pos = res;
}

void ImpProblem::init_item_weights(){
    item_w.resize(n, 0);
    for (ImpLong i = 0; i < m; i++) {
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {
            const ImpLong j = y->idx;
            item_w[j]++;
        }
    }

    for (ImpLong j = 0; j < n; j++) {
        item_w[j] = (item_w[j] > 0)? 1/item_w[j]: 1;
    }
}
