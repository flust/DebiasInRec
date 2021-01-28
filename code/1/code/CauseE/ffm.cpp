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
        const shared_ptr<ImpData> &d2, const shared_ptr<ImpData> &d1_treat,
        const shared_ptr<ImpData> &d2_treat) {
    const ImpLong Df1 = max(d1->Ds[fi], d1_treat->Ds[fi]);
    const ImpLong Df2 = max(d2->Ds[fj], d2_treat->Ds[fj]);

    const vector<Node*> &X1 = d1->Xs[fi];
    const vector<Node*> &X2 = d2->Xs[fj];
    const vector<Node*> &X1_treat = d1_treat->Xs[fi];
    const vector<Node*> &X2_treat = d2_treat->Xs[fj];

    
    init_mat(W_treat[f12], Df1, k);
    init_mat(H_treat[f12], Df2, k);
    
    P_treat[f12].resize(d1_treat->m*k, 0);
    Q_treat[f12].resize(d2_treat->m*k, 0);
    UTX(X1_treat, d1_treat->m, W_treat[f12], P_treat[f12]);
    UTX(X2_treat, d2_treat->m, H_treat[f12], Q_treat[f12]);

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

ImpDouble ImpProblem::calc_cross_treat(const ImpLong &i, const ImpLong &j) {
    ImpDouble cross_value = 0.0;
    for (ImpInt f1 = 0; f1 < fu; f1++) {
        for (ImpInt f2 = fu; f2 < f; f2++) {
            const ImpInt f12 = index_vec(f1, f2, f);
            const ImpDouble *pp_treat = P_treat[f12].data();
            const ImpDouble *qp_treat = Q_treat[f12].data();
            cross_value += inner(pp_treat+i*k, qp_treat+j*k, k);
        }
    }
    return cross_value;
}

void ImpProblem::init_y_tilde() {
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m; i++) {
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {
            ImpLong j = y->idx;
            y->val = calc_cross(i, j);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < n; j++) {
        for (YNode* y = V->Y[j]; y < V->Y[j+1]; y++) {
            ImpLong i = y->idx;
            y->val = calc_cross(i, j);
        }
    }

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m_treat; i++) {
        for (YNode* y = U_treat->Y[i]; y < U_treat->Y[i+1]; y++) {
            ImpLong j = y->idx;
            y->val = calc_cross_treat(i, j);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < n_treat; j++) {
        for (YNode* y = V_treat->Y[j]; y < V_treat->Y[j+1]; y++) {
            ImpLong i = y->idx;
            y->val = calc_cross_treat(i, j);
        }
    }
}

void ImpProblem::update_cross(const bool &sub_type, const Vec &S,
        const Vec &Q1, Vec &W1, const vector<Node*> &X12, Vec &P1, bool do_ctrl) {
    axpy( S.data(), W1.data(), S.size(), 1);
    
    shared_ptr<ImpData> U_, V_;
    ImpLong m_, n_;
    if(do_ctrl){
        U_ = U, V_ = V;
        m_ = m, n_ = n;
    }
    else{
        U_ = U_treat, V_ = V_treat;
        m_ = m_treat, n_ = n_treat;
    }

    const ImpLong m1 = (sub_type)? m_ : n_;

    shared_ptr<ImpData> U1 = (sub_type)? U_:V_;
    shared_ptr<ImpData> V1 = (sub_type)? V_:U_;

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

void ImpProblem::init() {
    lambda = param->lambda;
    ldiff = param->ldiff;
    wn = 1;

    c_norm = U->M.size();
    t_norm = U_treat->M.size();

    m = U->m;
    n = V->m;
    
    m_treat = U_treat->m;
    n_treat = V_treat->m;

    fu = U->f;
    fv = V->f;
    f = fu+fv;

    k = param->k;

    const ImpInt nr_blocks = f*(f+1)/2;

    W.resize(nr_blocks);
    H.resize(nr_blocks);

    P.resize(nr_blocks);
    Q.resize(nr_blocks);

    W_treat.resize(nr_blocks);
    H_treat.resize(nr_blocks);

    P_treat.resize(nr_blocks);
    Q_treat.resize(nr_blocks);

    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? U: V);
        const shared_ptr<ImpData> d1_treat = ((f1<fu)? U_treat: V_treat);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);
        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? U: V);
            const shared_ptr<ImpData> d2_treat = ((f2<fu)? U_treat: V_treat);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);
            const ImpInt f12 = index_vec(f1, f2, f);
            if( !param->self_side && ( f1 >= fu || f2 < fu ))
                continue;
            init_pair(f12, fi, fj, d1, d2, d1_treat, d2_treat);
        }
    }

    init_item_weights();
    init_y_tilde();

    init_L_pos();
    init_expyy();
}

void ImpProblem::gd_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, const Vec &Wreg, Vec &G, bool do_ctrl) {
    fill(G.begin(), G.end(), 0);
    axpy( W1.data(), G.data(), G.size(), lambda);
    axpy( W1.data(), G.data(), G.size(), ldiff);
    axpy( Wreg.data(), G.data(), G.size(), -ldiff);
    gd_pos_cross(f1, Q1, W1, G, do_ctrl);
}


void ImpProblem::hs_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V, Vec &Hv, const Vec &Q1,
        const vector<Node*> &X, const vector<YNode*> &Y, Vec &Hv_, const ImpDouble norm_) {
    hs_pos_cross(m1, n1, V, Hv, Q1, X, Y, Hv_, norm_);
}

void ImpProblem::cg(const ImpInt &f1, const ImpInt &f2, Vec &S1,
        const Vec &Q1, const Vec &G, Vec &P1, bool do_ctrl) {
    
    shared_ptr<ImpData> U_, V_;
    ImpLong m_, n_;
    ImpDouble norm_;
    if(do_ctrl){
        U_ = U, V_ = V;
        m_ = m, n_ = n;
        norm_ = c_norm;
    }
    else{
        U_ = U_treat, V_ = V_treat;
        m_ = m_treat, n_ = n_treat;
        norm_ = t_norm;
    }

    const ImpInt base = (f1 < fu)? 0: fu;
    const ImpInt fi = f1-base;

    const shared_ptr<ImpData> U1 = (f1 < fu)? U_:V_;
    const vector<YNode*> &Y = U1->Y;
    const vector<Node*> &X = U1->Xs[fi];

    const ImpLong m1 = (f1 < fu)? m_:n_;
    const ImpLong n1 = (f1 < fu)? n_:m_;

    const ImpLong Df1k = G.size();
    const ImpInt nr_threads = param->nr_threads;
    Vec Hv_(nr_threads*Df1k);

    ImpInt nr_cg = 0, max_cg = 20;
    ImpDouble g2 = 0, r2, cg_eps = 0.09, alpha = 0, beta = 0, gamma = 0, vHv;

    Vec V(Df1k, 0), R(Df1k, 0), Hv(Df1k, 0);

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

        axpy( V.data(), Hv.data(), V.size(), lambda + ldiff);
        hs_cross(m1, n1, V, Hv, Q1, X, Y, Hv_, norm_);

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
    const vector<Node*> &U1_treat = U_treat->Xs[f1], &V1_treat = V_treat->Xs[f2-fu];
    Vec &W1 = W[f12], &H1 = H[f12], &P1 = P[f12], &Q1 = Q[f12];
    Vec &W1_treat = W_treat[f12], &H1_treat = H_treat[f12], &P1_treat = P_treat[f12], &Q1_treat = Q_treat[f12];

    Vec GW(W1.size(),0), GH(H1.size(),0);
    Vec SW(W1.size(),0), SH(H1.size(),0);
    Vec GW_treat(W1_treat.size(),0), GH_treat(H1_treat.size(),0);
    Vec SW_treat(W1_treat.size(),0), SH_treat(H1_treat.size(),0);

    gd_cross(f1, Q1, W1, W1_treat, GW, true);
    cg(f1, f2, SW, Q1, GW, P1, true);
    line_search(f1, f2, SW, Q1, W1, W1_treat, P1, GW, true);
    update_cross(true, SW, Q1, W1, U1, P1, true);

    gd_cross(f1, Q1_treat, W1_treat, W1, GW_treat, false);
    cg(f1, f2, SW_treat, Q1_treat, GW_treat, P1_treat, false);
    line_search(f1, f2, SW_treat, Q1_treat, W1_treat, W1, P1_treat, GW_treat, false);
    update_cross(true, SW_treat, Q1_treat, W1_treat, U1_treat, P1_treat, false);
    
    gd_cross(f2, P1, H1, H1_treat, GH, true);
    cg(f2, f1, SH, P1, GH, Q1, true);
    line_search(f2, f1, SH, P1, H1, H1_treat, Q1, GH, true);
    update_cross(false, SH, P1, H1, V1, Q1, true);

    gd_cross(f2, P1_treat, H1_treat, H1, GH_treat, false);
    cg(f2, f1, SH_treat, P1_treat, GH_treat, Q1_treat, false);
    line_search(f2, f1, SH_treat, P1_treat, H1_treat, H1, Q1_treat, GH_treat, false);
    update_cross(false, SH_treat, P1_treat, H1_treat, V1_treat, Q1_treat, false);
}

void ImpProblem::one_epoch() {
    for (ImpInt f1 = 0; f1 < fu; f1++)
        for (ImpInt f2 = fu; f2 < f; f2++)
            solve_cross(f1, f2);
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

void ImpProblem::init_Pva_Qva_at_bt_treat(){
    for (ImpInt f1 = 0; f1 < f; f1++) {
        const shared_ptr<ImpData> d1 = ((f1<fu)? Uva: V_treat);
        const ImpInt fi = ((f1>=fu)? f1-fu: f1);

        for (ImpInt f2 = f1; f2 < f; f2++) {
            const shared_ptr<ImpData> d2 = ((f2<fu)? Uva: V_treat);
            const ImpInt fj = ((f2>=fu)? f2-fu: f2);

            const ImpInt f12 = index_vec(f1, f2, f);
            if( !param->self_side && ( f1 >= fu || f2 < fu ))
                continue;
            UTX(d1->Xs[fi], d1->m, W_treat[f12], Pva[f12]);
            UTX(d2->Xs[fj], d2->m, H_treat[f12], Qva[f12]);
        }
    }
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
            z[k] += pred_i_j(i, j);
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
            ImpDouble yy = (pred_i_j(i, j))*y->fid;
            if (-yy > 0)
                ploss += w2 *(-yy + log1p( exp(yy) ));
            else
                ploss += w2 * log1p( exp(-yy) );
        }
    }
    loss = ploss/Uva->M.size();
}

void ImpProblem::logloss_treat() {
    ImpDouble tr_loss_t = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: tr_loss_t)
    for(ImpLong i = 0; i < m_treat; i++){
        for(YNode *y = U_treat->Y[i]; y < U_treat->Y[i+1]; y++){

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
            ImpDouble yy = (pred_i_j(i, j))*y->fid;
            if (-yy > 0)
                ploss += w2 *(-yy + log1p( exp(yy) ));
            else
                ploss += w2 * log1p( exp(-yy) );
        }
    }
    loss = ploss/Uva->M.size();
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
        //init_Pva_Qva_at_bt_treat();
        //logloss_treat();
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
    const ImpDouble y_ij = y->fid, expyy = y->expyy;
    return  iw*w2 * -y_ij / (1 + expyy);
}

ImpDouble ImpProblem::l_pos_hessian(const YNode *y, const ImpDouble iw) {
    const ImpDouble w2 = (y->fid > 0)? 1 : wn;
    const ImpDouble expyy = y->expyy;
    return iw*w2 * expyy / (1 + expyy) / (1 + expyy);
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
    
#pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m_treat; i++) {
        for (YNode* y = U_treat->Y[i]; y < U_treat->Y[i+1]; y++) {
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong j = 0; j < n_treat; j++) {
        for (YNode* y = V_treat->Y[j]; y < V_treat->Y[j+1]; y++) {
            y->expyy = exp( y->val * (ImpDouble) y->fid);
        }
    }
}

void ImpProblem::gd_pos_cross(const ImpInt &f1, const Vec &Q1, const Vec &W1, Vec &G, bool do_ctrl) {
    shared_ptr<ImpData> U_, V_;
    ImpLong m_, n_;
    ImpDouble norm_;
    if(do_ctrl){
        U_ = U, V_ = V;
        m_ = m, n_ = n;
        norm_ = c_norm;
    }
    else{
        U_ = U_treat, V_ = V_treat;
        m_ = m_treat, n_ = n_treat;
        norm_ = t_norm;
    }

    const ImpLong &m1 = (f1 < fu)? m_:n_;
    const shared_ptr<ImpData> U1 = (f1 < fu)? U_:V_;
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
        axpy(G_.data()+i*block_size, G.data(), block_size, 1.0 / norm_ );
}

void ImpProblem::hs_pos_cross(const ImpLong &m1, const ImpLong &n1, const Vec &V, Vec &Hv, const Vec &Q1, const vector<Node*> &X, const vector<YNode*> &Y, Vec &Hv_, const ImpDouble norm_) {
   
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

                const ImpDouble iw = 1;

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
        axpy(Hv_.data()+i*block_size, Hv.data(), block_size, 1.0 / norm_);
}

void ImpProblem::line_search(const ImpInt &f1, const ImpInt &f2, Vec &S1, const Vec &Q1, const Vec &W1, const Vec &Wreg, Vec &P1, const Vec &G, bool do_ctrl) {

    shared_ptr<ImpData> U_, V_;
    ImpLong m_, n_;
    ImpDouble* L_pos_;
    ImpDouble norm_;
    if(do_ctrl){
        U_ = U, V_ = V;
        m_ = m, n_ = n;
        L_pos_ = &L_pos;
        norm_ = c_norm;
    }
    else{
        U_ = U_treat, V_ = V_treat;
        m_ = m_treat, n_ = n_treat;
        L_pos_ = &L_pos_treat;
        norm_ = t_norm;
    }

    const ImpInt base = (f1 < fu)? 0: fu;
    const ImpInt fi = f1-base;
    shared_ptr<ImpData> U1 = (f1 < fu)? U_:V_;
    vector<YNode*> &Y = U1->Y;
    const vector<Node*> &X = U1->Xs[fi];
    const ImpLong m1 = (f1 < fu)? m_:n_;

    ImpDouble sTg, wTs=0, sTs=0;
    sTg = inner(S1.data(), G.data(), S1.size());
    wTs = lambda*inner(S1.data(), W1.data(), S1.size()) + ldiff*( inner(S1.data(), W1.data(), S1.size()) - inner(S1.data(), Wreg.data(), S1.size()) );
    sTs = (lambda + ldiff)*inner(S1.data(), S1.data(), S1.size());

    Vec XS(P1.size(), 0);
    UTX(X, m1, S1, XS);
    calc_delta_y_cross(Y, m1, XS, Q1);
    
    ImpDouble theta = 1, beta = 0.5, nu = 0.1;
    while(true){
        if(theta < 1e-20){
            scal(S1.data(), S1.size(), 0);
            cerr << "Step size too small and skip this block." << endl;
            break;
        }
        ImpDouble L_pos_new = calc_L_pos(Y, m1, theta, norm_);
        ImpDouble delta = L_pos_new - *L_pos_ + (theta*wTs + 0.5*theta*theta*sTs);
        if( delta <= nu * theta * sTg ){
            *L_pos_ = L_pos_new;
            scal(S1.data(), S1.size(), theta);
            break;
        }
        theta *= beta;
    }
    if( theta != 1 )
        cerr << "Do line search " << theta << endl << flush;
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

ImpDouble ImpProblem::calc_L_pos(vector<YNode*> &Y, const ImpLong m1, const ImpDouble theta, const ImpDouble norm_){
    ImpDouble L_pos_new = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: L_pos_new)
    for(ImpLong i = 0; i < m1; i++){
        for(YNode *y = Y[i]; y != Y[i+1]; y++){
            const ImpDouble y_hat_new = y->val + theta * y->delta;
            const ImpDouble yy = y_hat_new * (ImpDouble) y->fid;
            const ImpDouble w2 = (y->fid > 0)? 1 : wn;

            const ImpLong idx = (m1 == m)? y->idx: i;
            const ImpDouble iw = param->item_weight? item_w[idx]: 1;

            if( -yy > 0 )
                L_pos_new += iw*w2 * (-yy + log1p( exp(yy) )) ;
            else
                L_pos_new += iw*w2 * log1p( exp(-yy) ) ;
        }
    }
    return L_pos_new / norm_;
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
                res += iw * w2 * (-yy + log1p( exp(yy) )) ;
            else
                res += iw * w2 * log1p( exp(-yy) ) ;
        }
    }
    L_pos = res / c_norm;
    
    ImpDouble res_treat = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: res_treat)
    for (ImpLong i = 0; i < m_treat; i++) {
        for (YNode* y = U_treat->Y[i]; y < U_treat->Y[i+1]; y++) {

            const ImpDouble iw = param->item_weight? item_w[y->idx]: 1;
            const ImpDouble w2 = (y->fid > 0)? 1 : wn;
            const ImpDouble yy = y->val * (ImpDouble) y->fid;

            if( -yy > 0 )
                res_treat += iw * w2 * (-yy + log1p( exp(yy) )) ;
            else
                res_treat += iw * w2 * log1p( exp(-yy) ) ;
        }
    }
    L_pos_treat = res_treat / t_norm;
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
