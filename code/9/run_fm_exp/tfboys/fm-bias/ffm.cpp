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

ImpDouble inner(const ImpDouble *p, const ImpDouble *q, const ImpInt k)
{
    return cblas_ddot(k, p, 1, q, 1);
}

ImpDouble tensor_product(const ImpDouble *w, const ImpDouble *h, const ImpDouble *z, const ImpInt d) {
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
    vec.resize(nr_rows*nr_cols);
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

void ImpProblem::solve_pos_bias(){
    Vec G(K, 0), S(K, 0), H(K, 0);

    Vec G_(nr_threads*K, 0), H_(nr_threads*K, 0);

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U->m; i++) {
        const ImpInt id = omp_get_thread_num();
        for (YNode* y = U->Y[i]; y < U->Y[i+1]; y++) {
            const ImpInt k = y->kdx; 
            G_[k+id*K] += l_pos_grad(y);
            H_[k+id*K] += l_pos_hessian(y);
        }
    }

    for(ImpInt i = 0; i < nr_threads; i++) {
        axpy(G_.data()+i*K, G.data(), K, 1);
        axpy(H_.data()+i*K, H.data(), K, 1);
    }

    for (ImpInt k = 0; k < K; k++)
        S[k] = -G[k]/H[k];

    ImpDouble sTg = inner(S.data(), G.data(), S.size());
    calc_delta_y_pos(U->Y, m, S);

    ImpDouble theta = 1, beta = 0.5, nu = 0.1;
    while(true){
        if(theta < 1e-20){
            scal(S.data(), S.size(), 0);
            cerr << "Step size too small and skip this block." << endl;
            break;
        }
        ImpDouble L_pos_new = calc_L_pos(U, theta);
        ImpDouble delta = L_pos_new - L_pos;
        if( delta <= nu * theta * sTg ){
            L_pos = L_pos_new;
            scal(S.data(), S.size(), theta);
            break;
        }
        theta *= beta;
    }
    if( theta != 1 )
        cerr << "Sub-problem: position bias did line search " << theta << endl << flush;
    update_pos_bias(S);
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

    sort(perm.begin(), perm.end(), sort_by_jdx);

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



//==============================================================
//==============================================================

void ImpProblem::UTx(const Node* x0, const Node* x1, const Vec &A, ImpDouble *c) {
    for (const Node* x = x0; x < x1; x++) {
        const ImpLong idx = x->idx;
        const ImpDouble val = x->val;
        for (ImpInt di = 0; di < d; di++) {
            ImpLong jd = idx*d+di;
            c[di] += val*A[jd];
        }
    }
}

void ImpProblem::UTX(const vector<Node*> &X, const ImpLong m1, const Vec &A, Vec &C) {
    fill(C.begin(), C.end(), 0);
    ImpDouble* c = C.data();
#pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < m1; i++)
        UTx(X[i], X[i+1], A, c+i*d);
}


void ImpProblem::init_params() {
    const ImpLong Dfu = U->D, Dfv = V->D;

    init_mat(E_u, Dfu, d);
    init_mat(E_v, Dfv, d);
    pos_b.resize(K, 0);

    W.resize(m*d, 0);
    H.resize(n*d, 0);
    if(is_solve_imp)
        ipb.resize(K, 0);

    UTX(U->X, m, E_u, W);
    UTX(V->X, n, E_v, H);

    if (w != 0)
        HTH.resize(d*d);
}



void ImpProblem::init_y_tilde_expyy() {
    calc_val_expyy(U);
    calc_val_expyy(V);

    if(!is_solve_imp) {
        init_ival(U);
        init_ival(V);
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
}

void ImpProblem::update_pos_bias(const Vec &S){
    axpy( S.data(), pos_b.data(), S.size(), 1);
    calc_val_expyy(U);
    calc_val_expyy(V);
}


void ImpProblem::init() {
    init_params();
    init_y_tilde_expyy();
    init_L_pos();
}


void ImpProblem::gd_cross(const shared_ptr<ImpData> &U1, const Vec &W1,
        const Vec &H1, const Vec &iW1, const Vec &iH1, const Vec &E1, Vec &G1) {
    fill(G1.begin(), G1.end(), 0);
    axpy(E1.data(), G1.data(), G1.size(), lambda);
    gd_pos_cross(U1, H1, G1);
    if (w != 0 && !is_solve_imp)
        gd_neg_cross(U1, W1, H1, iW1, iH1, G1);
}


void ImpProblem::hs_cross(const shared_ptr<ImpData> &U1, Vec &M,
        const Vec &H1, Vec &Hv, Vec &Hv_) {
    hs_pos_cross(U1, M, H1, Hv, Hv_);
    if (w != 0 && !is_solve_imp) {
        hs_neg_cross(U1, M, Hv, Hv_);
    }
}

void ImpProblem::cg(const shared_ptr<ImpData> &U1, const Vec &G1,
        const Vec &H1, Vec &S1) {

    const ImpLong Df = U1->D, Dfd = Df*d;
    Vec Hv_(nr_threads*Dfd);

    ImpInt nr_cg = 0, max_cg = 20;
    ImpDouble g2 = 0, r2, alpha = 0, beta = 0, gamma = 0, vHv, cg_eps = 0.09;

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

        axpy(M.data(), Hv.data(), M.size(), lambda);

        hs_cross(U1, M, H1, Hv, Hv_);

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
}

void ImpProblem::solve_block(const shared_ptr<ImpData> &U1, Vec &W1, const Vec &H1, const Vec &iW1, const Vec &iH1, Vec &E1){

    Vec G1(E1.size(),0);
    Vec S1(E1.size(),0);

    gd_cross(U1, W1, H1, iW1, iH1, E1, G1);
    cg(U1, G1, H1, S1);
    line_search(U1, E1, H1, G1, S1);
    update_cross(U1, S1, E1, W1);
    cout << "G:" << inner(G1.data(), G1.data(), G1.size()) << endl;
}


void ImpProblem::one_epoch() {
    
    solve_block(U, W, H, iW, iH, E_u);
    solve_block(V, H, W, iH, iW,E_v);

    if (is_solve_imp)
        solve_pos_bias();
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
    const ImpDouble *w_va_p =  W_va.data(), *hp = H.data();
    #pragma omp parallel for schedule(dynamic) 
    for (ImpLong i = 0; i < Uva->m; i++) {
        for(YNode* y = Uva->Y[i]; y < Uva->Y[i+1]; y++){
            const ImpLong i = y->idx, j = y->jdx, k = y->kdx;
            y->val = inner(w_va_p+i*d, hp+j*d, d) + pos_b[k];
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
    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        for (YNode* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            const ImpLong idx = y->idx, jdx = y->jdx, kdx = y->kdx;
            y->val = inner(W.data()+idx*d, H.data()+jdx*d, d) + pos_b[kdx];
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
            y->ival = inner(iW.data()+idx*d, iH.data()+jdx*d, d) + ipb[kdx];
        }
    }
}



void ImpProblem::gd_pos_cross(const shared_ptr<ImpData> &U1, const Vec &H1, Vec &G) {
    const vector<Node*> &X = U1->X;
    const vector<YNode*> &Y = U1->Y;

    const ImpLong block_size = G.size();
    Vec G_(nr_threads*block_size, 0);
    
    const ImpDouble *hp = H1.data();

    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        Vec pk(d, 0);
        const ImpInt id = omp_get_thread_num();
        for (YNode* y = Y[i]; y < Y[i+1]; y++) {
            const ImpLong j = (U1->pivot == 0)? y->jdx: y->idx;
            const ImpDouble *h1 = hp + j*d;
            const ImpDouble scale = l_pos_grad(y);
            axpy(h1, pk.data(), d, scale);
        }

        for (Node* x = X[i]; x < X[i+1]; x++) {
            const ImpLong idx = x->idx;
            const ImpDouble val = x->val;
            ImpDouble* Gp = G_.data() + id*block_size;
            axpy(pk.data(), Gp+idx*d, d, val);
        }
    }

    for(ImpInt i = 0; i < nr_threads; i++)
        axpy(G_.data()+i*block_size, G.data(), block_size, 1);
}

void ImpProblem::gd_neg_cross(const shared_ptr<ImpData> &U1, const Vec &W1,
        const Vec &H1, const Vec &iW1, const Vec &iH1, Vec &G) {

    const vector<Node*> &X = U1->X;
    const ImpLong n1 = H1.size()/d;

    Vec T(U1->m*d, 0), iHTH(d*d, 0), iT(U1->m*d, 0), In(n, 1), oH(d, 0);

    mm(H1.data(), H1.data(), HTH.data(), d, n1);
    mm(W1.data(), HTH.data(), T.data(), U1->m, d, d, 1);

    mm(iH1.data(), H1.data(), iHTH.data(), d, n1);
    mm(iW1.data(), iHTH.data(), iT.data(), U1->m, d, d, 1);

    mv(H.data(), In.data(), oH.data(), n, d, 0, true);

    const ImpLong block_size = G.size();

    Vec G_(nr_threads*block_size, 0);

    const ImpDouble b_sum = sum(ipb);
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
                G_[jd+id*block_size] += w*(K*(t1[di]-it[di])-b_sum*oH[di])*val;
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

void ImpProblem::hs_pos_cross(const shared_ptr<ImpData> U1, Vec &M,
        const Vec H1, Vec &Hv, Vec &Hv_) {
    
    fill(Hv_.begin(), Hv_.end(), 0);
    const ImpDouble *hp = H1.data();

    const ImpLong block_size = Hv.size();


    #pragma omp parallel for schedule(dynamic)
    for (ImpLong i = 0; i < U1->m; i++) {
        const ImpInt id = omp_get_thread_num();

        Vec phi(d, 0), ka(d, 0);
        UTx(U1->X[i], U1->X[i+1], M, phi.data());

        for (YNode* y = U1->Y[i]; y < U1->Y[i+1]; y++) {
            const ImpLong j = (U1->pivot == 0)? y->jdx: y->idx;

            const ImpDouble *dh = hp + j*d;
            const ImpDouble val = inner(phi.data(), dh, d) * l_pos_hessian(y);

            for (ImpInt di = 0; di < d; di++)
                ka[di] += val*dh[di];
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
        axpy(Hv_.data()+i*block_size, Hv.data(), block_size, 1);
}

void ImpProblem::hs_neg_cross(const shared_ptr<ImpData> U1, Vec &M, Vec &Hv, Vec &Hv_) {

    Vec MHZ(U1->D*d, 0);
    mm(M.data(), HTH.data(), MHZ.data(), U1->D, d, d);

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
        axpy(Hv_.data()+i*block_size, Hv.data(), block_size, 1);
}

void ImpProblem::line_search(const shared_ptr<ImpData> &U1, const Vec &E, const Vec &H1, 
        const Vec &G, Vec &S1) {

    const vector<Node*> &X = U1->X;

    const ImpLong Df = U1->D, Dfd = Df*d;
    Vec Hs_(nr_threads*Dfd);

    ImpDouble sTg_neg=0, sHs=0, sTg, wTs=0, sTs=0;

    sTg = inner(S1.data(), G.data(), S1.size());
    wTs = lambda*inner(S1.data(), E.data(), S1.size());
    sTs = lambda*inner(S1.data(), S1.data(), S1.size());

    if (w != 0) {
        sTg_neg = inner(S1.data(), Gneg.data(), S1.size());

        Vec Hs(Dfd, 0);
        fill(Hs_.begin(), Hs_.end(), 0);
        hs_neg_cross(U1, S1, Hs, Hs_);
        sHs = inner(S1.data(), Hs.data(), S1.size());
    }

    Vec XS(U1->m*d, 0);
    UTX(X, U1->m, S1, XS);
    calc_delta_y_cross(U1, XS, H1);
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
    if( theta != 1 )
        cerr << "Sub-problem:" << U1->pivot << " did line search " << theta << endl << flush;
}

void ImpProblem::calc_delta_y_cross(const shared_ptr<ImpData> &U1, const Vec &XS,
        const Vec &H1){
    const ImpDouble *hp = H1.data();
    const ImpDouble *pp = XS.data();
    #pragma omp parallel for schedule(dynamic)
    for(ImpLong i = 0 ; i < U1->m; i++){
        for(YNode *y = U1->Y[i]; y != U1->Y[i+1]; y++){
            const ImpLong j = (U1->pivot == 0)? y->jdx: y->idx;
            y->delta = inner(pp + i*d, hp + j*d, d);
        }
    }
}

void ImpProblem::calc_delta_y_pos(vector<YNode*> &Y, const ImpLong m1, const Vec &S){
    #pragma omp parallel for schedule(dynamic)
    for(ImpLong i = 0 ; i < m1; i++){
        for(YNode *y = Y[i]; y != Y[i+1]; y++){
            const ImpInt pos = y->kdx;
            y->delta = S[pos];
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
    return L_pos_new;
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
    L_pos = res;
}

void ImpProblem::input_whz(const shared_ptr<ImpData> &U1, const Vec &E_u1,
        const Vec &iH1, const Vec &ipb1) {
    iW.resize(U1->m *d);
    UTX(U1->X, U1->m, E_u1, iW);
    iH = iH1;
    ipb = ipb1;
}

void ImpProblem::echo_pos_avg(const ImpLong &j) {
   Vec pos_avg(K, 0), ratio(K-1, 0); ImpDouble val = 0; 
   const ImpDouble *wp =  W.data(), *hp = H.data();
   for (ImpLong i = 0; i < m; i++)  {
       for (ImpLong k = 0; k < K; k++) {
            val = inner(wp+i*d, hp+j*d, d) + pos_b[k];
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

void ImpProblem::save_npy_files(){
    update_Pva_Yva();

    const ImpLong shape_p[] = {mt,  d};
    const ImpLong shape_q[] = {n,  d};

    npy::SaveArrayAsNumpy("./Pva", false, 2, shape_p, W_va);
    npy::SaveArrayAsNumpy("./Qva", false, 2, shape_q, H);
}
