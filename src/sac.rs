use nalgebra as na;
use rand::seq::IteratorRandom;
use crate::model::Model;

/// A RANSAC wrapper for P3P algorithms
pub fn ransac(world_points: &na::Matrix3xX<f64>, bearing_vectors: &na::Matrix3xX<f64>,
              solver: &dyn Fn(&na::Matrix3<f64>, &na::Matrix3<f64>) -> Vec<Model>,
              max_iterations: u32,
              inlier_dist_threshold: f64,
              probability: f64) -> Option<Model> {

    // parameters
    let mut iterations: u32 = 0;
    let mut best_inliers_count: u32 = 0;
    let mut k: f64 = 1.0;

    let mut skipped_count: u32 = 0;

    let max_skip: u32 = max_iterations * 10;

    let mut rng = rand::thread_rng();

    let mut p_w = na::Matrix3::<f64>::zeros();
    let mut p_i = na::Matrix3::<f64>::zeros();
    let indices: Vec<usize> = (0..world_points.shape().1).collect();
    // sample 4 points (3 pts for p3p, 1 pt for choosing models
    let mut sampled_indices: Vec<&usize> = indices.iter().choose_multiple(&mut rng, 4);

    let mut best_model: Option<Model> = None;

    while (iterations < k as u32) & (iterations < max_iterations) & (skipped_count < max_skip) {
        p_w.set_column(0, &world_points.column(*sampled_indices[0]));
        p_w.set_column(1, &world_points.column(*sampled_indices[1]));
        p_w.set_column(2, &world_points.column(*sampled_indices[2]));
        p_i.set_column(0, &bearing_vectors.column(*sampled_indices[0]));
        p_i.set_column(1, &bearing_vectors.column(*sampled_indices[1]));
        p_i.set_column(2, &bearing_vectors.column(*sampled_indices[2]));

        // compute model
        let models = solver(&p_w, &p_i);

        // solution returns a vector of models
        // skip if empty
        if models.len() == 0 {
            skipped_count += 1;
            continue;
        }

        // select model with the smallest reprojection cost
        let mut selected_model_idx = 0;
        let mut min_cost: f64 = f64::INFINITY;
        let pt_idx = *sampled_indices[3];
        for i in 0..models.len() {
            let cost = models[i].reprojection_dist_of(world_points.column(pt_idx), bearing_vectors.column(pt_idx));
            if cost < min_cost {
                selected_model_idx = i;
                min_cost = cost;
            }
        }

        // count the number of inliers within a distance threshold to the model
        let dists_to_all = models[selected_model_idx].reprojection_dists_of(&world_points,
                                                                            &bearing_vectors);
        let mut inlier_count = 0;
        for i in 0..dists_to_all.shape().1 {
            if dists_to_all[i] < inlier_dist_threshold {
                inlier_count += 1;
            }
        }

        // we have a better hypothesis
        if inlier_count > best_inliers_count {
            best_inliers_count = inlier_count;
            best_model.insert(models[selected_model_idx]);

            // Compute the k parameter (k=log(z)/log(1-w^n))
            let w: f64 = (best_inliers_count as f64) / (world_points.shape().1 as f64);
            let p_no_outliers: f64 = {
                let mut p = 1.0 - w.powi(sampled_indices.len() as i32);
                p = p.max(f64::EPSILON);
                p = p.min(1.0 - f64::EPSILON);
                p
            };

            // Avoid division by 0.
            k = (1.0 - probability).ln() / (p_no_outliers).ln();
        }

        // resample
        sampled_indices = indices.iter().choose_multiple(&mut rng, 4);
        iterations += 1;
    }

    return best_model;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p3p::{kneip, fischler, grunert};
    use nalgebra as na;
    use rand::Rng;

    #[test]
    fn test_ransac() {
        let mut rng = rand::thread_rng();
        let t_gt: na::Vector3<f64> = rng.gen();
        let r_gt: na::Rotation3<f64> = rng.gen();
        let rotation_gt: na::Matrix3<f64> = r_gt.matrix().into_owned();

        // generate random points with outliers
        let n_points = 10;
        let p_src = na::Matrix3xX::<f64>::new_random(n_points);
        let p_tgt: na::Matrix3xX<f64> = {
            let mut p_tgt: na::Matrix3xX<f64> = rotation_gt * &p_src;
            for (i, mut col) in p_tgt.column_iter_mut().enumerate() {
                col += t_gt;
            }
            // add outliers
            let n_outliers = 1;
            for i in 0..n_outliers {
                let mut new_column: na::Vector3<f64> = p_tgt.column(i).into();
                new_column[0] += 10.0 * (i as f64 + 1.0);
                new_column[1] -= 10.0 * (i as f64 + 1.0);
                p_tgt.set_column(i, &new_column);
            }
            // normalize p_tgt to be bearing vectors
            for (i, mut col) in p_tgt.column_iter_mut().enumerate() {
                col /= col.norm();
            }
            p_tgt
        };

        // test with ransac
        let result = ransac(&p_src, &p_tgt, &fischler, 50, 0.1, 0.95);
        match result {
            Some(x) => {
                assert!(x.rotation.relative_eq(&rotation_gt, 1e-7, 1e-7));
                assert!(x.translation.relative_eq(&t_gt, 1e-7, 1e-7));
            }
            None => assert!(false, "RANSAC failed."),
        }
    }
}