/// Structural Causal Model (SCM) Engine & Causal Discovery (PC Algorithm).
///
/// Implements Bayesian conditioning (observe), do-calculus interventions (intervene),
/// 3-step counterfactual logic (Abduction-Action-Prediction), and skeleton/orientation PC algorithm.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct CausalModel {
    pub names: Vec<String>,
    pub name_to_idx: HashMap<String, usize>,
    /// Adjacency weight matrix (parents -> children)
    /// weights[i][j] is the causal weight of i on j.
    pub weights: Vec<Vec<f64>>,
    /// Exogenous noise variances
    pub noise_variances: Vec<f64>,
    /// Exogenous noise means
    pub noise_means: Vec<f64>,
}

impl CausalModel {
    pub fn new(
        names: Vec<String>,
        weights: Vec<Vec<f64>>,
        noise_variances: Vec<f64>,
        noise_means: Vec<f64>,
    ) -> Self {
        let mut name_to_idx = HashMap::new();
        for (i, name) in names.iter().enumerate() {
            name_to_idx.insert(name.clone(), i);
        }
        Self {
            names,
            name_to_idx,
            weights,
            noise_variances,
            noise_means,
        }
    }

    /// Helper to compute (I - W^T)^-1
    fn get_inv_i_minus_wt(&self) -> Option<Vec<Vec<f64>>> {
        let n = self.names.len();
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            mat[i][i] = 1.0;
            for j in 0..n {
                // transpose: W^T is weights[j][i]
                mat[i][j] -= self.weights[j][i];
            }
        }
        invert_matrix(&mat)
    }

    /// Compute the joint mean vector and covariance matrix of X.
    /// X = (I - W^T)^-1 * U
    pub fn compute_joint_distribution(&self) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
        let n = self.names.len();
        let inv = self.get_inv_i_minus_wt()?;

        // Mean: mu_X = inv * mu_U
        let mut mean = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                mean[i] += inv[i][j] * self.noise_means[j];
            }
        }

        // Covariance: Sigma_X = inv * Sigma_U * inv^T
        // Since Sigma_U is diagonal with noise_variances:
        // Sigma_X[i][j] = sum_k (inv[i][k] * noise_variances[k] * inv[j][k])
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += inv[i][k] * self.noise_variances[k] * inv[j][k];
                }
                cov[i][j] = sum;
            }
        }

        Some((mean, cov))
    }

    /// Condition via Bayes Rule: observe(var, val)
    /// Returns the conditional mean and standard deviation for all variables.
    pub fn observe(&self, evidence: &HashMap<String, f64>) -> Option<HashMap<String, (f64, f64)>> {
        let (mu, sigma) = self.compute_joint_distribution()?;
        let n = self.names.len();

        let mut evidence_indices = Vec::new();
        let mut evidence_values = Vec::new();
        for (name, &val) in evidence {
            if let Some(&idx) = self.name_to_idx.get(name) {
                evidence_indices.push(idx);
                evidence_values.push(val);
            }
        }

        if evidence_indices.is_empty() {
            // No evidence, return joint stats
            let mut results = HashMap::new();
            for i in 0..n {
                let std = sigma[i][i].sqrt();
                results.insert(self.names[i].clone(), (mu[i], std));
            }
            return Some(results);
        }

        // Partition into Query (Q) and Evidence (E)
        let mut q_indices = Vec::new();
        for i in 0..n {
            if !evidence_indices.contains(&i) {
                q_indices.push(i);
            }
        }

        // Submatrices
        // Sigma_EE
        let num_e = evidence_indices.len();
        let mut sig_ee = vec![vec![0.0; num_e]; num_e];
        for i in 0..num_e {
            for j in 0..num_e {
                sig_ee[i][j] = sigma[evidence_indices[i]][evidence_indices[j]];
            }
        }
        let sig_ee_inv = invert_matrix(&sig_ee)?;

        // Conditional mean and var for each query variable
        let mut results = HashMap::new();
        for &q in &q_indices {
            // Sigma_qE: 1 x num_e
            let mut sig_qe = vec![0.0; num_e];
            for i in 0..num_e {
                sig_qe[i] = sigma[q][evidence_indices[i]];
            }

            // sig_qe * sig_ee_inv
            let mut coeff = vec![0.0; num_e];
            for i in 0..num_e {
                for j in 0..num_e {
                    coeff[i] += sig_qe[j] * sig_ee_inv[j][i];
                }
            }

            // mu_q|E = mu_q + coeff * (x_E - mu_E)
            let mut cond_mean = mu[q];
            for i in 0..num_e {
                cond_mean += coeff[i] * (evidence_values[i] - mu[evidence_indices[i]]);
            }

            // var_q|E = Sigma_qq - sig_qe * sig_ee_inv * sig_Eq
            let mut cond_var = sigma[q][q];
            let mut sum_deduction = 0.0;
            for i in 0..num_e {
                sum_deduction += coeff[i] * sigma[evidence_indices[i]][q];
            }
            cond_var -= sum_deduction;
            let cond_std = cond_var.max(0.0).sqrt();

            results.insert(self.names[q].clone(), (cond_mean, cond_std));
        }

        // Include evidence variables as deterministic results
        for (name, &val) in evidence {
            results.insert(name.clone(), (val, 0.0));
        }

        Some(results)
    }

    /// Intervene: do(var = val) cuts incoming edges and propagates values forward.
    pub fn intervene(&self, intervention: &HashMap<String, f64>) -> Option<HashMap<String, (f64, f64)>> {
        let n = self.names.len();
        // Create an intervened model
        let mut int_weights = self.weights.clone();
        let mut int_noise_vars = self.noise_variances.clone();
        let mut int_noise_means = self.noise_means.clone();

        for (name, &val) in intervention {
            if let Some(&idx) = self.name_to_idx.get(name) {
                // Cut incoming edges: weights[*][idx] = 0
                for r in 0..n {
                    int_weights[r][idx] = 0.0;
                }
                // Set structural equation of idx to be deterministic: X_idx = val + 0*U
                int_noise_means[idx] = val;
                int_noise_vars[idx] = 1e-15; // essentially zero variance
            }
        }

        let int_model = CausalModel::new(
            self.names.clone(),
            int_weights,
            int_noise_vars,
            int_noise_means,
        );
        int_model.compute_joint_distribution().map(|(mu, sigma)| {
            let mut results = HashMap::new();
            for i in 0..n {
                let std = sigma[i][i].sqrt();
                results.insert(self.names[i].clone(), (mu[i], std));
            }
            results
        })
    }

    /// Counterfactual Logic: Abduction, Action, Prediction
    pub fn counterfactual(
        &self,
        evidence: &HashMap<String, f64>,
        intervention: &HashMap<String, f64>,
        query: &[String],
    ) -> Option<HashMap<String, f64>> {
        let n = self.names.len();
        // Step 1: Abduction. Infer exogenous noise values U given evidence.
        // We know U = (I - W^T) X.
        // We want to compute the conditional expectation of U given X_E = x_E.
        // Let's find the joint distribution of [U; X] which is Gaussian.
        // Mean of U is self.noise_means.
        // Covariance of U is diagonal Sigma_U.
        // Covariance of U and X: Cov(U, X) = Cov(U, (I - W^T)^-1 * U) = Sigma_U * (I - W^T)^-T.
        let inv = self.get_inv_i_minus_wt()?;
        let (mu_x, sigma_x) = self.compute_joint_distribution()?;

        let mut evidence_indices = Vec::new();
        let mut evidence_values = Vec::new();
        for (name, &val) in evidence {
            if let Some(&idx) = self.name_to_idx.get(name) {
                evidence_indices.push(idx);
                evidence_values.push(val);
            }
        }

        let u_posterior = if evidence_indices.is_empty() {
            self.noise_means.clone()
        } else {
            // We use conditional expectation: E[U | X_E = x_E]
            // E[U | X_E] = mu_U + Cov(U, X_E) * Cov(X_E, X_E)^-1 * (x_E - mu_E)
            let num_e = evidence_indices.len();
            // Cov(U, X_E) is an n x num_e matrix.
            // Cov(U, X)_ij = Sigma_U[i][i] * inv[j][i]
            let mut cov_u_xe = vec![vec![0.0; num_e]; n];
            for i in 0..n {
                for (j_idx, &e_col) in evidence_indices.iter().enumerate() {
                    cov_u_xe[i][j_idx] = self.noise_variances[i] * inv[e_col][i];
                }
            }

            // Cov(X_E, X_E)^-1
            let mut sig_ee = vec![vec![0.0; num_e]; num_e];
            for i in 0..num_e {
                for j in 0..num_e {
                    sig_ee[i][j] = sigma_x[evidence_indices[i]][evidence_indices[j]];
                }
            }
            let sig_ee_inv = invert_matrix(&sig_ee)?;

            // E[U | X_E] computation
            let mut u_mean = self.noise_means.clone();
            for i in 0..n {
                // Row i of cov_u_xe multiplied by sig_ee_inv
                let mut coeff = vec![0.0; num_e];
                for j in 0..num_e {
                    for k in 0..num_e {
                        coeff[j] += cov_u_xe[i][k] * sig_ee_inv[k][j];
                    }
                }

                let mut diff = 0.0;
                for j in 0..num_e {
                    diff += coeff[j] * (evidence_values[j] - mu_x[evidence_indices[j]]);
                }
                u_mean[i] += diff;
            }
            u_mean
        };

        // Step 2 & 3: Action & Prediction.
        // Intervene on the structural equations by setting X_I = x_I.
        // And propagate the posterior noise values u_posterior.
        // Solve (I - W'^T) X = U', where U'_i = x_i if i in I, and U'_i = u_posterior_i if i not in I.
        // Also W'_ji = W_ji if i not in I, and 0 if i in I.
        let mut int_weights = self.weights.clone();
        let mut u_prime = u_posterior.clone();

        for (name, &val) in intervention {
            if let Some(&idx) = self.name_to_idx.get(name) {
                for r in 0..n {
                    int_weights[r][idx] = 0.0;
                }
                u_prime[idx] = val;
            }
        }

        // Solve: X_i = sum_j (int_weights[j][i] * X_j) + u_prime[i]
        // This is (I - W'^T) X = U'. So X = (I - W'^T)^-1 * U'.
        let mut mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            mat[i][i] = 1.0;
            for j in 0..n {
                mat[i][j] -= int_weights[j][i];
            }
        }
        let int_inv = invert_matrix(&mat)?;

        let mut x_counterfactual = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                x_counterfactual[i] += int_inv[i][j] * u_prime[j];
            }
        }

        let mut results = HashMap::new();
        for name in query {
            if let Some(&idx) = self.name_to_idx.get(name) {
                results.insert(name.clone(), x_counterfactual[idx]);
            }
        }
        Some(results)
    }
}

// ═══════════════════════════════════════════
//  Causal Discovery: PC Algorithm
// ═══════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct PCResult {
    /// 1.0 indicates a directed edge from i -> j, 0.5 indicates an undirected edge, 0.0 is no edge.
    pub adjacency: Vec<Vec<f64>>,
    /// Confidence or strength of correlation for the edges.
    pub confidences: Vec<Vec<f64>>,
    pub names: Vec<String>,
}

/// Run PC Algorithm for Causal Discovery
pub fn discover(data: &[Vec<f64>], names: Vec<String>, alpha: f64) -> PCResult {
    let n_vars = names.len();
    let n_samples = data.len();

    // 1. Compute empirical correlation matrix
    let mut means = vec![0.0; n_vars];
    for row in data {
        for j in 0..n_vars {
            means[j] += row[j];
        }
    }
    for j in 0..n_vars {
        means[j] /= n_samples as f64;
    }

    let mut cov = vec![vec![0.0; n_vars]; n_vars];
    for row in data {
        for i in 0..n_vars {
            for j in 0..n_vars {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }
    for i in 0..n_vars {
        for j in 0..n_vars {
            cov[i][j] /= (n_samples - 1) as f64;
        }
    }

    // Undirected graph skeleton initialized to complete graph (except self-loops)
    let mut adj = vec![vec![true; n_vars]; n_vars];
    for i in 0..n_vars {
        adj[i][i] = false;
    }

    // Separating sets
    let mut sepsets = HashMap::<(usize, usize), Vec<usize>>::new();

    // Skeleton discovery
    let mut depth = 0;
    loop {
        let mut stable = true;
        for i in 0..n_vars {
            for j in 0..n_vars {
                if !adj[i][j] {
                    continue;
                }

                // Get neighbors of i excluding j
                let mut neighbors = Vec::new();
                for k in 0..n_vars {
                    if adj[i][k] && k != j {
                        neighbors.push(k);
                    }
                }

                if neighbors.len() < depth {
                    continue;
                }

                stable = false;

                let combinations = get_combinations(&neighbors, depth);
                for cond_set in combinations {
                    // Test conditional independence of i and j given cond_set
                    let p_val = test_conditional_independence(&cov, i, j, &cond_set, n_samples);
                    if p_val > alpha {
                        adj[i][j] = false;
                        adj[j][i] = false;
                        sepsets.insert((i, j), cond_set.clone());
                        sepsets.insert((j, i), cond_set.clone());
                        break;
                    }
                }
            }
        }

        if stable {
            break;
        }
        depth += 1;
        if depth >= n_vars {
            break;
        }
    }

    // Output representation: directed graph. Initialize as undirected (0.5 in both directions)
    // if adj is true.
    let mut res_adj = vec![vec![0.0; n_vars]; n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            if adj[i][j] {
                res_adj[i][j] = 0.5;
            }
        }
    }

    // 2. Orient V-structures (colliders): i - k - j where i and j are not adjacent
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i >= j || adj[i][j] {
                continue;
            }
            // Look for common neighbors k
            for k in 0..n_vars {
                if adj[i][k] && adj[j][k] {
                    // Check separating set
                    let in_sepset = if let Some(set) = sepsets.get(&(i, j)) {
                        set.contains(&k)
                    } else {
                        false
                    };
                    if !in_sepset {
                        // Orient: i -> k <- j
                        res_adj[i][k] = 1.0;
                        res_adj[k][i] = 0.0;
                        res_adj[j][k] = 1.0;
                        res_adj[k][j] = 0.0;
                    }
                }
            }
        }
    }

    // 3. Meek's orientation rules
    loop {
        let mut changed = false;

        // Rule 1: Orient k - j into k -> j if there is i -> k and i, j are not adjacent
        for i in 0..n_vars {
            for k in 0..n_vars {
                if res_adj[i][k] == 1.0 {
                    for j in 0..n_vars {
                        if res_adj[k][j] == 0.5 && res_adj[j][k] == 0.5 && !adj[i][j] && i != j {
                            res_adj[k][j] = 1.0;
                            res_adj[j][k] = 0.0;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Rule 2: Orient i - j into i -> j if there is a chain i -> k -> j
        for i in 0..n_vars {
            for k in 0..n_vars {
                if res_adj[i][k] == 1.0 {
                    for j in 0..n_vars {
                        if res_adj[k][j] == 1.0 && res_adj[i][j] == 0.5 && res_adj[j][i] == 0.5 {
                            res_adj[i][j] = 1.0;
                            res_adj[j][i] = 0.0;
                            changed = true;
                        }
                    }
                }
            }
        }

        // Rule 3: Orient i - j into i -> j if there are two chains i - k -> j and i - l -> j with k, l not adjacent
        for i in 0..n_vars {
            for j in 0..n_vars {
                if res_adj[i][j] == 0.5 && res_adj[j][i] == 0.5 {
                    for k in 0..n_vars {
                        for l in 0..n_vars {
                            if k != l && !adj[k][l] &&
                               res_adj[i][k] == 0.5 && res_adj[k][i] == 0.5 && res_adj[k][j] == 1.0 &&
                               res_adj[i][l] == 0.5 && res_adj[l][i] == 0.5 && res_adj[l][j] == 1.0
                            {
                                res_adj[i][j] = 1.0;
                                res_adj[j][i] = 0.0;
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        if !changed {
            break;
        }
    }

    // Compute edge confidences based on absolute partial correlation (1 - p_value)
    let mut confidences = vec![vec![0.0; n_vars]; n_vars];
    for i in 0..n_vars {
        for j in 0..n_vars {
            if adj[i][j] {
                // Retrieve the separating set if they were independent, else empty set
                let set = sepsets.get(&(i, j)).cloned().unwrap_or(vec![]);
                let p_val = test_conditional_independence(&cov, i, j, &set, n_samples);
                confidences[i][j] = (1.0 - p_val).max(0.0).min(1.0);
            }
        }
    }

    PCResult {
        adjacency: res_adj,
        confidences,
        names,
    }
}

/// Compute combinations of size `k` from `n` items
fn get_combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    let mut results = Vec::new();
    let mut combo = vec![0; k];
    combinations_recurse(items, k, 0, 0, &mut combo, &mut results);
    results
}

fn combinations_recurse(
    items: &[usize],
    k: usize,
    depth: usize,
    next: usize,
    combo: &mut [usize],
    results: &mut Vec<Vec<usize>>,
) {
    if depth == k {
        results.push(combo.to_vec());
        return;
    }
    for i in next..items.len() {
        combo[depth] = items[i];
        combinations_recurse(items, k, depth + 1, i + 1, combo, results);
    }
}

/// Test conditional independence using partial correlation & Fisher's z-transform
fn test_conditional_independence(
    cov: &[Vec<f64>],
    i: usize,
    j: usize,
    cond: &[usize],
    n_samples: usize,
) -> f64 {
    if cond.is_empty() {
        // Zero-order correlation
        let denom = (cov[i][i] * cov[j][j]).sqrt();
        if denom.abs() < 1e-12 {
            return 1.0; // Independent if variance is 0
        }
        let r = cov[i][j] / denom;
        return fisher_z_test(r, 0, n_samples);
    }

    let idx_a = vec![i, j];
    let idx_b = cond.to_vec();

    // Sigma_AA (2 x 2)
    let sig_aa = vec![
        vec![cov[idx_a[0]][idx_a[0]], cov[idx_a[0]][idx_a[1]]],
        vec![cov[idx_a[1]][idx_a[0]], cov[idx_a[1]][idx_a[1]]],
    ];

    // Sigma_BB (d x d)
    let d = idx_b.len();
    let mut sig_bb = vec![vec![0.0; d]; d];
    for r in 0..d {
        for c in 0..d {
            sig_bb[r][c] = cov[idx_b[r]][idx_b[c]];
        }
    }

    let sig_bb_inv = match invert_matrix(&sig_bb) {
        Some(inv) => inv,
        None => return 1.0, // Treat as independent if conditioning covariance is singular
    };

    // Sigma_AB (2 x d)
    let mut sig_ab = vec![vec![0.0; d]; 2];
    for r in 0..2 {
        for c in 0..d {
            sig_ab[r][c] = cov[idx_a[r]][idx_b[c]];
        }
    }

    // Sigma_BA (d x 2)
    let mut sig_ba = vec![vec![0.0; 2]; d];
    for r in 0..d {
        for c in 0..2 {
            sig_ba[r][c] = cov[idx_b[r]][idx_a[c]];
        }
    }

    // Sigma_A|B = Sigma_AA - Sigma_AB * Sigma_BB^-1 * Sigma_BA
    // sig_ab * sig_bb_inv (2 x d)
    let mut tmp = vec![vec![0.0; d]; 2];
    for r in 0..2 {
        for c in 0..d {
            for k in 0..d {
                tmp[r][c] += sig_ab[r][k] * sig_bb_inv[k][c];
            }
        }
    }

    // tmp * sig_ba (2 x 2)
    let mut deduct = vec![vec![0.0; 2]; 2];
    for r in 0..2 {
        for c in 0..2 {
            for k in 0..d {
                deduct[r][c] += tmp[r][k] * sig_ba[k][c];
            }
        }
    }

    let sig_a_cond_b = vec![
        vec![sig_aa[0][0] - deduct[0][0], sig_aa[0][1] - deduct[0][1]],
        vec![sig_aa[1][0] - deduct[1][0], sig_aa[1][1] - deduct[1][1]],
    ];

    let denom = (sig_a_cond_b[0][0] * sig_a_cond_b[1][1]).sqrt();
    if denom.abs() < 1e-12 {
        return 1.0;
    }
    let r_partial = sig_a_cond_b[0][1] / denom;
    fisher_z_test(r_partial, d, n_samples)
}

fn fisher_z_test(r: f64, cond_len: usize, n_samples: usize) -> f64 {
    // Clip r to avoid division by zero or log of negative
    let r_clipped = r.max(-0.999999).min(0.999999);
    let z = 0.5 * ((1.0 + r_clipped) / (1.0 - r_clipped)).ln();
    let df = (n_samples as f64 - cond_len as f64 - 3.0).max(1.0);
    let stat = z.abs() * df.sqrt();
    // 2 * (1 - Phi(stat))
    2.0 * (1.0 - std_normal_cdf(stat))
}

fn std_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * (-x * x).exp();
    sign * y
}

// ═══════════════════════════════════════════
//  Matrix Helpers
// ═══════════════════════════════════════════

fn invert_matrix(m: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = m.len();
    let mut a = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = m[i][j];
        }
        a[i][n + i] = 1.0;
    }
    for i in 0..n {
        // Find pivot
        let mut pivot_row = i;
        for r in i+1..n {
            if a[r][i].abs() > a[pivot_row][i].abs() {
                pivot_row = r;
            }
        }
        if a[pivot_row][i].abs() < 1e-12 {
            return None; // Singular
        }
        if pivot_row != i {
            a.swap(i, pivot_row);
        }
        let pivot = a[i][i];
        for j in i..2*n {
            a[i][j] /= pivot;
        }
        for r in 0..n {
            if r != i {
                let factor = a[r][i];
                for j in i..2*n {
                    a[r][j] -= factor * a[i][j];
                }
            }
        }
    }
    let mut inv = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            inv[i][j] = a[i][n + j];
        }
    }
    Some(inv)
}
