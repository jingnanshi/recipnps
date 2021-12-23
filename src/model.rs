use nalgebra as na;
use nalgebra::U3;

pub struct Model {
    pub rotation: na::Matrix3<f64>,
    pub translation: na::Vector3<f64>,
}

impl Model {
    /// Calculate the reprojection cost of a given pair of bearing vector
    /// and world point correspondences.
    ///
    /// We assume:
    /// p_cam = R_gt p_world + t_gt
    fn reprojection_cost(&self, world_point: na::MatrixSlice3x1<f64>,
                         bearing_vector: na::MatrixSlice3x1<f64>) -> f64{
        // we assume:
        // p_cam = R_gt p_world + t_gt
        let reprojection = (self.rotation * world_point + self.translation).normalize();

        // this is equivalent to
        // 1 - cos(angle between the two vectors)
        // and the range will be from 0 to 2
        return 1.0 - reprojection.dot(&bearing_vector);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra as na;
    use rand::Rng;

    fn test_reprojection_cost() {
        let mut rng = rand::thread_rng();
        let t_gt: na::Vector3<f64> = rng.gen();
        let r_gt: na::Rotation3<f64> = rng.gen();
        let rotation_gt: na::Matrix3<f64> = r_gt.matrix().into_owned();

        let p_src = na::Matrix3::<f64>::new_random();
        let mut p_tgt: na::Matrix3<f64> = rotation_gt * p_src;
        for (i, mut col) in p_tgt.column_iter_mut().enumerate() {
            col += t_gt;
        }
        let mut model_to_test = Model { rotation: rotation_gt, translation: t_gt };
        model_to_test.reprojection_cost(p_src.column(0), p_tgt.column(0));
    }
}
