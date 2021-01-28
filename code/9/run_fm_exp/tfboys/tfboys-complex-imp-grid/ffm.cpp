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

void print_vec(const Vec &v) {
    for (ImpDouble val: v)
        cout << val << ",";
    cout << endl;
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

const ImpDouble inner(const ImpDouble *p, const ImpDouble *q, const ImpInt k)
{
    return cblas_ddot(k, p, 1, q, 1);
}

const ImpDouble logit(const ImpDouble &ctr){
    return log( ctr / (1.0 - ctr) );
}

const ImpDouble tensor_product(const ImpDouble *w, const ImpDouble *h, const ImpDouble *z, const ImpInt d) {
    ImpDouble prod_val = 0;
    for (ImpInt di = 0; di < d; di++){
        prod_val += w[di]*h[di]*z[di];
    }
    return prod_val;
}

void hadamard_product(const ImpDouble *a, const ImpDouble *b, const ImpLong size, ImpDouble *c){
    for (ImpInt i = 0; i < size; i++) {
        c[i] = a[i]*b[i];
    }
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

void ImpData::read(bool has_label, const ImpLong Ds) {
    ifstream fs(file_name);
    string line, label_block, label_str;
    char dummy;

    ImpLong idx, ips, field, y_nnz=0, x_nnz=0;
    ImpInt label;
    ImpDouble val;

    while (getline(fs, line)) {
        m++;
        istringstream iss(line);

        if (has_label) {
            iss >> label_block;
            istringstream labelst(label_block);
            ImpInt pos_counter = 0;
            while (getline(labelst, label_str, ',')) {
                istringstream labels(label_str);
                labels >> idx >> dummy >> label >> dummy >> ips;
                y_nnz++;
                pos_counter++;
            }
            K = max(K, pos_counter);
        }

        while (iss >> field >> dummy >> idx >> dummy >> val) {
            if (Ds != 0 && Ds <= idx)
                continue;
            D = max(D, idx+1);
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
            ImpInt pos = 0;
            while (getline(labelst, label_str, ',')) {
                nnz_j++;
                istringstream labels(label_str);
                labels >> idx >> dummy >> label >> dummy >> ips;

                M[nnz_j-1].label = (label > 0) ? 1 : -1;

                M[nnz_j-1].idx = i;
                M[nnz_j-1].jdx = idx;
                M[nnz_j-1].kdx = pos;

                pos++;
            }
            nny[i] = nnz_j;
        }

        while (iss >> field >> dummy >> idx >> dummy >> val) {
            if (Ds != 0 && Ds <= idx)
                continue;
            nnz_i++;
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

void ImpData::decode_onehot() {
    N.resize(m);
    X.resize(m+1);
    Y.resize(m+1);

    X[0] = N.data();
    for (ImpLong i = 0; i < m; i++){
        N[i].idx = i;
        N[i].val = 1;
        X[i+1] = N.data()+(i+1);
    }
}

void ImpData::normalize() {
#pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m; i++) {
        ImpInt nnz = X[i+1] - X[i];
        for (Node *x = X[i]; x < X[i+1]; x++) {
            x->val = x->val / nnz;
        }
    }
}

void ImpData::transY(const vector<YNode*> &YT) {
    ImpLong n = YT.size() - 1;
    vector<pair<ImpLong, YNode*>> perm;
    ImpLong nnz = 0;
    vector<ImpLong> nnzs(m, 0);

    for (ImpLong i = 0; i < n; i++) {
        for (YNode* y = YT[i]; y < YT[i+1]; y++) {
            const ImpLong id = (pivot == 1)? y->jdx: y->kdx;
            if (id >= m )
              continue;
            nnzs[id]++;
            perm.emplace_back(i, y);
            nnz++;
        }
    }

    auto sort_by_jdx = [&] (const pair<ImpLong, YNode*> &lhs,
            const pair<ImpLong, YNode*> &rhs) {
        return tie(lhs.second->jdx, lhs.first) < tie(rhs.second->jdx, rhs.first);
    };

    auto sort_by_kdx = [&] (const pair<ImpLong, YNode*> &lhs,
            const pair<ImpLong, YNode*> &rhs) {
        return tie(lhs.second->kdx, lhs.first) < tie(rhs.second->kdx, rhs.first);
    };

    if (pivot == 1)
        sort(perm.begin(), perm.end(), sort_by_jdx);
    else
        sort(perm.begin(), perm.end(), sort_by_kdx);

    M.resize(nnz);
    nnz_y = nnz;
    for (ImpLong nnz_i = 0; nnz_i < nnz; nnz_i++) {
        M[nnz_i].idx = perm[nnz_i].second->idx;
        M[nnz_i].jdx = perm[nnz_i].second->jdx;

        M[nnz_i].kdx = perm[nnz_i].second->kdx;
        M[nnz_i].label = perm[nnz_i].second->label;
    }

    Y[0] = M.data();
    ImpLong start_idx = 0;
    for (ImpLong i = 0; i < m; i++) {
        start_idx += nnzs[i];
        Y[i+1] = M.data()+start_idx;
    }
}

void ImpData::dummy_position() {
    for (ImpLong i = 0; i < m; i++)
        N[i].idx = 0;
}



//==============================================================
//==============================================================

void ImpProblem::UTx(const Node* x0, const Node* x1, const Vec &A, ImpDouble *c) {
    for (const Node* x = x0; x < x1; x++) {
        const ImpLong idx = x->idx;
        const ImpDouble val = x->val;
        for (ImpInt di = 0; di < d; di++) {
            ImpLong jd = idx*d+di;
            assert( jd < A.size());
            c[di] += val*A[jd];
        }
    }
}

void ImpProblem::UTX(const vector<Node*> &X, const ImpLong m1, const Vec &A, Vec &C) {
    fill(C.begin(), C.end(), 0);
    ImpDouble* c = C.data();
    assert(m1 < X.size());
    assert(m1*d - 1 < C.size());
#pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m1; i++){
        UTx(X[i], X[i+1], A, c+i*d);
    }
}


void ImpProblem::init_params() {
    const ImpLong Dfu = U->D, Dfv = V->D, Dfp = P->D;

    init_mat(E_u, Dfu, d);
    init_mat(E_v, Dfv, d);
    init_mat(E_p, Dfp, d);

    W.resize(m*d, 0);
    H.resize(n*d, 0);
    Z.resize(K*d, 0);

    UTX(U->X, m, E_u, W);
    UTX(V->X, n, E_v, H);
    UTX(P->X, K, E_p, Z);

    if (w != 0) {
        HTHZTZ.resize(d*d, 0);
        iHTHiZTZ.resize(d*d, 0);
    }
}



void ImpProblem::init_y_tilde_expyy() {
    calc_val_expyy(U);
    calc_val_expyy(V);
    calc_val_expyy(P);

    if (!solve_imp) {
        init_ival(U);
        init_ival(V);
        init_ival(P);
    }
}

void ImpProblem::update_cross(const shared_ptr<ImpData> U1, const Vec &S, Vec &E, Vec &W1) {
    axpy( S.data(), E.data(), S.size(), 1);
    const ImpLong m1 = U1->m;

    Vec XS(W1.size(), 0);
    UTX(U1->X, m1, S, XS);
    axpy( XS.data(), W1.data(), W1.size(), 1);

    calc_val_expyy(U);
    calc_val_expyy(V);
    calc_val_expyy(P);
}


void ImpProblem::init() {
    init_params();
    init_y_tilde_expyy();
    init_L_pos();
}


void ImpProblem::gd_cross(const shared_ptr<ImpData> &U1, const Vec &W1,
        const Vec &H1, const Vec &Z1, const Vec &iW1, const Vec &iH1,
        const Vec &iZ1, const Vec &E1, Vec &G1) {
    fill(G1.begin(), G1.end(), 0);
    if (U1->pivot == 2) {
        axpy(E1.data(), G1.data(), G1.size(), lambda);
    }
    else {
        axpy(E1.data(), G1.data(), G1.size(), lambda);
    }
    gd_pos_cross(U1, H1, Z1, G1);
    if (w != 0)
        gd_neg_cross(U1, W1, H1, Z1, iW1, iH1, iZ1, G1);
}


void ImpProblem::hs_cross(const shared_ptr<ImpData> &U1, Vec &M,
        const Vec &H1, const Vec &Z1, Vec &Hv, Vec &Hv_) {
    hs_pos_cross(U1, M, H1, Z1, Hv, Hv_);
    if (w != 0) {
        hs_neg_cross(U1, M, Hv, Hv_);
    }
}

void ImpProblem::cg(const shared_ptr<ImpData> &U1, const Vec &G1,
        const Vec &H1, const Vec &Z1, Vec &S1) {

    const ImpLong Df = U1->D, Dfd = Df*d;
    Vec Hv_(nr_threads*Dfd);

    ImpInt nr_cg = 0, max_cg = 20;
    ImpDouble g2 = 0, r2, alpha = 0, beta = 0, gamma = 0, vHv, cg_eps;
    if (U1->pivot == 2) {
        cg_eps = 0.01;
    } else {
        cg_eps = 0.09;
    }



    Vec M(Dfd, 0), R(Dfd, 0), Hv(Dfd, 0);

    for (ImpLong jd = 0; jd < Dfd; jd++) {
        R[jd] = -G1[jd];
        M[jd] = R[jd];
        g2 += G1[jd]*G1[jd];
    }

    r2 = g2;
    while (g2*cg_eps < r2 && nr_cg < max_cg) {
        nr_cg++;

        fill(Hv.begin(), Hv.end(), 0);
        fill(Hv_.begin(), Hv_.end(), 0);

        if (U1->pivot == 2) {
            axpy(M.data(), Hv.data(), M.size(), lambda);
        }
        else {
            axpy(M.data(), Hv.data(), M.size(), lambda);
        }

        hs_cross(U1, M, H1, Z1, Hv, Hv_);

        vHv = inner(M.data(), Hv.data(), Dfd);
        gamma = r2;
        alpha = gamma/vHv;
        axpy(M.data(), S1.data(), Dfd, alpha);
        axpy(Hv.data(), R.data(), Dfd, -alpha);
        r2 = inner(R.data(), R.data(), Dfd);
        beta = r2/gamma;
        scal(M.data(), Dfd, beta);
        axpy(R.data(), M.data(), Dfd, 1);
    }
#ifdef DEBUG
    cerr << U1->pivot << ":" << nr_cg << endl;
#endif
}


void ImpProblem::solve_block(const shared_ptr<ImpData> &U1, Vec &W1, const Vec &H1,
        const Vec &Z1, const Vec &iW1, const Vec &iH1, const Vec &iZ1, Vec &E1){
    Vec G1(E1.size(), 0);
    Vec S1(E1.size(), 0);
    
    gd_cross(U1, W1, H1, Z1, iW1, iH1, iZ1, E1, G1);
    cg(U1, G1, H1, Z1, S1);
    line_search(U1, E1, H1, Z1, G1, S1);
    update_cross(U1, S1, E1, W1);
#ifdef DEBUG
    if(!solve_imp)
        cerr << "G " << inner(G1.data(), G1.data(), G1.size()) << endl << flush;
#endif
}

void ImpProblem::one_epoch() {
    solve_block(U, W, H, Z, iW, iH, iZ, E_u);
    solve_block(V, H, W, Z, iH, iW, iZ, E_v);
    solve_block(P, Z, W, H, iZ, iW, iH, E_p);
}

void ImpProblem::init_va() {

    if (Uva->file_name.empty())
        return;

    mt = Uva->m;

    W_va.resize(mt*d);
    UTX(Uva->X, mt, E_u, W_va);

    cout << "iter";
    cout.width(13);
    cout << "tr_loss";
    cout.width(13);
    cout << "te_loss";
    cout.width(11);
    cout << "AUC";
    cout << endl;
}


void ImpProblem::calc_gauc(){
    ImpDouble gauc_sum = 0;
    ImpDouble gauc_weight_sum = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: gauc_sum, gauc_weight_sum)
    for (ImpLong i = 0; i < Uva->m; i++){
        ImpLong num_obv = ImpLong(Uva->Y[i+1] - Uva->Y[i]);
        Vec z(num_obv, 0), label(num_obv, 0);
        ImpLong k = 0;
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++, k++){
            z[k] = y->val;
            label[k] = y->label;
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
            z[k] = y->val;
            label[k] = y->label;
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

void ImpProblem::update_Pva_Yva() {
    UTX(Uva->X, mt, E_u, W_va);
    const ImpDouble *w_va_p =  W_va.data(), *hp = H.data(), *zp = Z.data();
    #pragma omp parallel for schedule(dynamic) 
    for (ImpLong i = 0; i < Uva->m; i++) {
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            const ImpLong i = y->idx, j = y->jdx, k = y->kdx;
            y->val = tensor_product(w_va_p+i*d, hp+j*d, zp+k*d, d);
        }
    }
}

void ImpProblem::logloss() {
    ImpDouble tr_loss_t = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: tr_loss_t)
    for(ImpLong i = 0; i < m; i++){
        for(YNode *y = U->Y[i]; y < U->Y[i+1]; y++){

            const ImpDouble yy = y->val * (ImpDouble) y->label;

            if( -yy > 0 )
                tr_loss_t += (-yy + log1p( exp(yy) ));
            else
                tr_loss_t += log1p( exp(-yy) );
        }
    }
    tr_loss = tr_loss_t/U->M.size();

    ImpDouble ploss = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: ploss)
    for (ImpLong i = 0; i < Uva->m; i++) {
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            ImpDouble yy = y->val * (ImpDouble) y->label;
            if (-yy > 0)
                ploss += (-yy + log1p( exp(yy) ));
            else
                ploss += log1p( exp(-yy) );
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

ImpDouble ImpProblem::calc_gauc_i(const Vec &z, const ImpLong &i, bool do_sum_all){
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
            if(y->idx == idx && y->label > 0){
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

ImpDouble ImpProblem::calc_auc_i(const Vec &z, const Vec &label){
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


void ImpProblem::print_epoch_info(ImpInt t) {
    cout.width(2);
    cout << t+1 << " ";
    if (!Uva->file_name.empty() && (t+1) % 1 == 0){
        update_Pva_Yva();
        logloss();
        calc_auc();
        cout.width(13);
        cout << setprecision(3) << tr_loss;
        cout.width(13);
        cout << setprecision(3) << loss;
        cout.width(13);
        cout << setprecision(3) << auc;
        cout << endl;
    }
}

void ImpProblem::solve() {
    init_va();
    for (ImpInt iter = 0; iter < nr_pass; iter++) {
        one_epoch();
        print_epoch_info(iter);
    }
}




ImpDouble ImpProblem::l_pos_grad(const YNode *y) {
    const ImpDouble y_ij = y->label, y_hat = y->val, expyy = y->expyy, ival = y->ival;
    return  -y_ij * expyy - w * (y_hat - ival);
}

ImpDouble ImpProblem::l_pos_hessian(const YNode *y) {
    const ImpDouble expyy = y->expyy;
    return expyy*(1-expyy) - w;
}

void ImpProblem::calc_val_expyy(const shared_ptr<ImpData> &U1){
#ifdef DEBUG
    cerr << "check_update" << U1->m << " " << U1->Y[U1->m] - U1->Y[0] <<  endl;
#endif
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        for (YNode* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            const ImpLong idx = y->idx, jdx = y->jdx, kdx = y->kdx;
            y->val = tensor_product(W.data()+idx*d, H.data()+jdx*d, Z.data()+kdx*d, d);
            if (y->val * (ImpDouble) y->label > 0 ) {
                y->expyy = exp( -y->val * (ImpDouble) y->label);
                y->expyy = y->expyy/(1+y->expyy);
            }
            else {
                y->expyy = exp(y->val * (ImpDouble) y->label);
                y->expyy = 1/(1+y->expyy);
            }
        }
    }
}

void ImpProblem::init_ival(const shared_ptr<ImpData> &U1){
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        for (YNode* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            const ImpLong idx = y->idx, jdx = y->jdx, kdx = y->kdx;
            assert( iW.size() > idx * d);
            assert( iH.size() > jdx * d);
            assert( iZ.size() > kdx * d);
            y->ival = tensor_product(iW.data()+idx*d, iH.data()+jdx*d, iZ.data()+kdx*d, d);
        }
    }
}



void ImpProblem::gd_pos_cross(const shared_ptr<ImpData> &U1, const Vec &H1, const Vec &Z1, Vec &G) {
    const vector<Node*> &X = U1->X;
    const vector<YNode*> &Y = U1->Y;

    const ImpLong block_size = G.size();
    Vec G_(nr_threads*block_size, 0);
    
    const ImpDouble *hp = H1.data();
    const ImpDouble *zp = Z1.data();

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        Vec pk(d, 0);
        const ImpInt id = omp_get_thread_num();
        for (YNode* y = Y[i]; y < Y[i+1]; y++) {
            const ImpLong j = (U1->pivot == 0)? y->jdx: y->idx;
            const ImpLong k = (U1->pivot == 2)? y->jdx: y->kdx;
            const ImpDouble *h1 = hp + j*d;
            const ImpDouble *z1 = zp + k*d;

            const ImpDouble scale = l_pos_grad(y);
            for (ImpInt di = 0; di < d; di++)
                pk[di] += scale*h1[di]*z1[di];
        }

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt di = 0; di < d; di++) {
                const ImpLong jd = idx*d+di;
                G_[jd+id*block_size] += pk[di]*val;
            }
        }
    }
    for(ImpInt i = 0; i < nr_threads; i++)
        axpy(G_.data()+i*block_size, G.data(), block_size, 1.0/norm_base);
}

void ImpProblem::gd_neg_cross(const shared_ptr<ImpData> &U1, const Vec &W1,
        const Vec &H1, const Vec &Z1, const Vec &iW1, const Vec &iH1, const Vec &iZ1, Vec &G) {

    const vector<Node*> &X = U1->X;
    const ImpLong n1 = H1.size()/d, K1 = Z1.size()/d;

    Vec HTH(d*d, 0), ZTZ(d*d, 0), T(U1->m*d, 0), iHTH(d*d, 0), iZTZ(d*d, 0), iT(U1->m*d, 0);

    mm(H1.data(), H1.data(), HTH.data(), d, n1);
    mm(Z1.data(), Z1.data(), ZTZ.data(), d, K1);

    hadamard_product(HTH.data(), ZTZ.data(), d*d, HTHZTZ.data());
    mm(W1.data(), HTHZTZ.data(), T.data(), U1->m, d, d, 1);

    mm(iH1.data(), H1.data(), iHTH.data(), d, n1);
    mm(iZ1.data(), Z1.data(), iZTZ.data(), d, K1);

    hadamard_product(iHTH.data(), iZTZ.data(), d*d, iHTHiZTZ.data());
    mm(iW1.data(), iHTHiZTZ.data(), iT.data(), U1->m, d, d, 1);

    const ImpLong block_size = G.size();

    Vec G_(nr_threads*block_size, 0);

    const ImpDouble *tp = T.data(), *ip = iT.data();

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        const ImpInt id = omp_get_thread_num();
        const ImpDouble *t1 = tp+i*d;
        const ImpDouble *it = ip+i*d;
        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt di = 0; di < d; di++) {
                const ImpLong jd = idx*d+di;
                G_[jd+id*block_size] += w*(t1[di]-it[di])*val;
            }
        }
    }

    Gneg.resize(G.size());
    fill(Gneg.begin(), Gneg.end(), 0);
    for(ImpInt i = 0; i < nr_threads; i++){
        axpy(G_.data()+i*block_size, G.data(), block_size, 1.0 / norm_base);
        axpy(G_.data()+i*block_size, Gneg.data(), block_size, 1.0 / norm_base);
    }
}

void ImpProblem::hs_pos_cross(const shared_ptr<ImpData> U1, Vec &M,
        const Vec H1, const Vec Z1, Vec &Hv, Vec &Hv_) {
    
    fill(Hv_.begin(), Hv_.end(), 0);
    const ImpDouble *hp = H1.data();
    const ImpDouble *zp = Z1.data();

    const ImpLong block_size = Hv.size();


    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        const ImpInt id = omp_get_thread_num();

        Vec phi(d, 0), ka(d, 0);
        UTx(U1->X[i], U1->X[i+1], M, phi.data());

        for (YNode* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            const ImpLong j = (U1->pivot == 0)? y->jdx: y->idx;
            const ImpLong k = (U1->pivot == 2)? y->jdx: y->kdx;

            const ImpDouble *dh = hp + j*d;
            const ImpDouble *dz = zp + k*d;

            const ImpDouble val = tensor_product(phi.data(), dh, dz, d) * l_pos_hessian(y);

            for (ImpInt di = 0; di < d; di++)
                ka[di] += val*dh[di]*dz[di];
        }

        for (Node* x = U1->X[i]; x < U1->X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt di = 0; di < d; di++) {
                const ImpLong jd = idx*d+di;
                Hv_[jd+id*block_size] += ka[di]*val;
            }
        }
    }

    for(ImpInt i = 0; i < nr_threads; i++)
        axpy(Hv_.data()+i*block_size, Hv.data(), block_size, 1.0 / norm_base);
}

void ImpProblem::hs_neg_cross(const shared_ptr<ImpData> U1, Vec &M, Vec &Hv, Vec &Hv_) {

    Vec MHZ(U1->D*d, 0);
    mm(M.data(), HTHZTZ.data(), MHZ.data(), U1->D, d, d);

    fill(Hv_.begin(), Hv_.end(), 0);
    const ImpLong block_size = Hv.size();


    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        const ImpInt id = omp_get_thread_num();
        Vec tau(d, 0);
        UTx(U1->X[i], U1->X[i+1], MHZ, tau.data());

        for (Node* x = U1->X[i]; x < U1->X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            for (ImpInt di = 0; di < d; di++) {
                const ImpLong jd = idx*d+di;
                Hv_[jd+id*block_size] += w*tau[di]*val;
            }
        }
    }

    for(ImpInt i = 0; i < nr_threads; i++)
        axpy(Hv_.data()+i*block_size, Hv.data(), block_size, 1.0 / norm_base);
}

void ImpProblem::line_search(const shared_ptr<ImpData> &U1, const Vec &E, const Vec &H1, 
        const Vec &Z1, const Vec &G, Vec &S1) {

    const vector<Node*> &X = U1->X;

    const ImpLong Df = U1->D, Dfd = Df*d;
    Vec Hs_(nr_threads*Dfd);

    ImpDouble sTg_neg=0, sHs=0, sTg, wTs=0, sTs=0;

    sTg = inner(S1.data(), G.data(), S1.size());
    ImpDouble this_lambda;
    if (U1->pivot == 2)
        this_lambda = lambda;
    else
        this_lambda = lambda;
    wTs = this_lambda*inner(S1.data(), E.data(), S1.size());
    sTs = this_lambda*inner(S1.data(), S1.data(), S1.size());

    if (w != 0) {
        sTg_neg = inner(S1.data(), Gneg.data(), S1.size());

        Vec Hs(Dfd, 0);
        fill(Hs_.begin(), Hs_.end(), 0);
        hs_neg_cross(U1, S1, Hs, Hs_);
        sHs = inner(S1.data(), Hs.data(), S1.size());
    }

    Vec XS(U1->m*d, 0);
    UTX(X, U1->m, S1, XS);
    calc_delta_y_cross(U1, XS, H1, Z1);
    ImpDouble theta = 1, beta = 0.5, nu = 0.1;
    while(true){
        if(theta < 1e-20){
            scal(S1.data(), S1.size(), 0);
            cerr << "Step size too small and skip this block." << endl;
            break;
        }
        ImpDouble L_pos_new = calc_L_pos(U1, theta);
        ImpDouble delta = L_pos_new - L_pos + theta * sTg_neg + 0.5 * theta * theta * sHs + (theta*wTs + 0.5*theta*theta*sTs);
        if( delta <= nu * theta * sTg ){
            L_pos = L_pos_new;
            scal(S1.data(), S1.size(), theta);
            break;
        }
        theta *= beta;
    }
#ifdef DEBUG
    if( theta != 1 )
        cerr << "Sub-problem:" << U1->pivot << " did line search " << theta << endl << flush;
#endif
}

void ImpProblem::calc_delta_y_cross(const shared_ptr<ImpData> &U1, const Vec &XS,
        const Vec &H1, const Vec &Z1){
    const ImpDouble *hp = H1.data();
    const ImpDouble *zp = Z1.data();
    const ImpDouble *pp = XS.data();
    #pragma omp parallel for schedule(dynamic)
    for(ImpLong i = 0 ; i < U1->m; i++){
        for(YNode *y = U1->Y[i]; y != U1->Y[i+1]; y++){
            const ImpLong j = (U1->pivot == 0)? y->jdx: y->idx;
            const ImpLong k = (U1->pivot == 2)? y->jdx: y->kdx;
            y->delta = tensor_product(pp + i*d, hp + j*d, zp + k*d, d);
        }
    }
}


ImpDouble ImpProblem::calc_L_pos(const shared_ptr<ImpData> &U1, const ImpDouble theta){
    ImpDouble L_pos_new = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: L_pos_new)
    for(ImpLong i = 0; i < U1->m; i++){
        for(YNode *y = U1->Y[i]; y != U1->Y[i+1]; y++){
            const ImpDouble y_hat_new = y->val + theta * y->delta, yy = y_hat_new * (ImpDouble) y->label, ival = y->ival;

            if( -yy > 0 )
                L_pos_new += (-yy + log1p( exp(yy) )) - 0.5 * w * (y_hat_new - ival) * (y_hat_new - ival);
            else
                L_pos_new += log1p( exp(-yy) ) - 0.5 * w * (y_hat_new - ival) * (y_hat_new - ival);
        }
    }
    return L_pos_new / norm_base;
}
    
void ImpProblem::init_L_pos(){
    ImpDouble res = 0;
    #pragma omp parallel for schedule(dynamic) reduction(+: res)
    for (ImpLong i = 0; i < m; i++) {
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {

            const ImpDouble yy = y->val * (ImpDouble) y->label, ival = y->ival;

            if( -yy > 0 )
                res += (-yy + log1p( exp(yy) )) - 0.5 * w * (y->val - ival) * (y->val - ival);
            else
                res += log1p( exp(-yy) ) - 0.5 * w * (y->val - ival) * (y->val - ival);
        }
    }
    L_pos = res / norm_base;
}

void ImpProblem::input_whz(const shared_ptr<ImpData> &U1, Vec &iE_u, Vec &iH1, Vec &iZ1) {
    iW.resize(U1->m *d);
    iH.resize(iH1.size());
    iZ.resize(iZ1.size());

    UTX(U1->X, U1->m, iE_u, iW);
    iH = iH1;
    iZ = iZ1;
}

void ImpProblem::init_imp_r_whz(const Vec &imp_r){
    print_vec(imp_r);
    iW.resize(m*d, 1.0 / sqrt(d) );
    iH.resize(n*d, 1.0 / sqrt(d) );
    iZ.resize(K*d, 0.0);

    for(ImpLong i = 0; i < K; i++)
        fill(iZ.begin() + i*d, iZ.begin() + (i+1)*d, imp_r[i]);
    
}


void ImpProblem::echo_pos_avg(const ImpLong &j) {
   Vec pos_avg(K, 0), ratio(K-1, 0); ImpDouble val = 0; 
   const ImpDouble *wp =  W.data(), *hp = H.data(), *zp = Z.data();
   for (ImpLong i = 0; i < m; i++)  {
       for (ImpLong k = 0; k < K; k++) {
            val = tensor_product(wp+i*d, hp+j*d, zp+k*d, d);
            if (val > 0)
                pos_avg[k] += 1/(1+exp(-val));
            else
                pos_avg[k] += exp(val)/(exp(val)+1);
       }
   }
   for (ImpLong k = 1; k < K; k++) {
       ratio[k-1] = pos_avg[k]/pos_avg[k-1];
   }
   print_vec(ratio);
}

void ImpProblem::calc_imp_r(){
    imp_r.resize(P->m, 0);
    fill(imp_r.begin(), imp_r.end(), 0);

#pragma omp parallel for schedule(dynamic)
    for(ImpLong k = 0; k < P->m; k++){
        for(YNode* y = P->Y[k]; y < P->Y[k+1]; y++){
            if(y->label == 1)
                imp_r[k]++;
        }
        imp_r[k] /= m;

        if(imp_r[k] > 0 )
            imp_r[k] = logit(imp_r[k]);
        else
            imp_r[k] = -20;
    }
}

void ImpProblem::calc_single_imp_r(){
    imp_r.resize(P->m, 0);
    fill(imp_r.begin(), imp_r.end(), 0);

    ImpDouble nr_p = 0;
#pragma omp parallel for schedule(dynamic), reduction(+:nr_p)
    for(ImpLong k = 0; k < P->m; k++){
        for(YNode* y = P->Y[k]; y < P->Y[k+1]; y++){
            if(y->label == 1)
                nr_p++;
        }
    }

    for(ImpLong k = 0; k < P->m; k++){
        if(nr_p > 0 )
            imp_r[k] = logit(nr_p / (m*P->m));
        else
            imp_r[k] = -20;
    }
}

void test_output(string file_name, Vec & A, ImpLong m, ImpLong n){
    ofstream of;
    of.open(file_name);
    for(ImpLong i = 0; i < m; i++){
        for(ImpLong j = 0; j < n; j++)
            of << A[i*n + j] << " ";
        of << endl;
    }
    of.close();
}

void ImpProblem::save_npy_files(){
    update_Pva_Yva();

    const ImpLong shape_p[] = {mt,  d};
    const ImpLong shape_q[] = {n,  d};

    npy::SaveArrayAsNumpy("./Pva", false, 2, shape_p, W_va);
    npy::SaveArrayAsNumpy("./Qva", false, 2, shape_q, H);
}
