use clam::{Metric, Number};

/// Dynamic Time Warping distance metric
#[derive(Debug)]
pub struct DynamicTimeWarping<T: Number, U: Number> {
    child_metric: Box<dyn Metric<T, U>>,
}

impl<T: Number, U: Number> DynamicTimeWarping<T, U> {
    pub fn new(child_metric: Box<dyn Metric<T, U>>) -> Self {
        Self { child_metric }
    }
}

impl<T: Number, U: Number> Metric<T, U> for DynamicTimeWarping<T, U> {
    fn name(&self) -> String {
        String::from("dtw")
    }

    fn one_to_one(&self, x: &[T], y: &[T]) -> U {
        let mut cost_matrix: Vec<Vec<U>> = Vec::with_capacity(y.len());

        // Construct the cost matrix
        for r in 0..y.len() {
            cost_matrix.push(Vec::with_capacity(x.len()));
            for c in 0..x.len() {
                // Compute the distance between X[c] and Y[r], corresponding to
                // the additional cost for this element in the cost matrix
                let dist: U = self.child_metric.one_to_one(&[x[c]], &[y[r]]);

                // Compute the cost for element (r, c) in the cost matrix
                // Calculated as the distance between X[c] and Y[r] plus the
                // minimum cost amongust the top, left, and top-left immediate
                // neighbors.   ඞ
                let mut neighbor_costs: Vec<U> = Vec::with_capacity(3);
                if r > 0 {
                    neighbor_costs.push(cost_matrix[r - 1][c]); // Top
                }
                if c > 0 {
                    neighbor_costs.push(cost_matrix[r][c - 1]); // Left
                }
                if c > 0 && r > 0 {
                    neighbor_costs.push(cost_matrix[r - 1][c - 1]); // Top-left
                }

                // Identify the minimum cost out of all the neighbor costs
                let min_neighbor_cost = neighbor_costs
                    .into_iter()
                    .reduce(|a, b| if a < b { a } else { b })
                    .unwrap_or(U::zero());

                // Update the cost matrix
                cost_matrix[r].push(dist + min_neighbor_cost);
            }
        }

        // Return the full distance between the two inputs
        cost_matrix[y.len() - 1][x.len() - 1]
    }

    fn is_expensive(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn my_test() {
        let child_metric = clam::metric_from_name::<f64, f32>("manhattan", false).unwrap();
        let dtw: DynamicTimeWarping<f64, f32> = DynamicTimeWarping::new(child_metric);
        let x = &[1.0, 3.0, 9.0, 2.0, 1.0];
        let y = &[2.0, 0.0, 0.0, 8.0, 7.0, 2.0];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 9.0);
    }

    #[test]
    fn my_test_2() {
        let child_metric = clam::metric_from_name::<i16, u8>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<i16, u8> = DynamicTimeWarping::new(child_metric);
        let x = &[2, 3, 1, 8, 5, 5, 6];
        let y = &[8, 7, 9, 2];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 27);
    }

    #[test]
    fn my_test_3() {
        let child_metric = clam::metric_from_name::<i16, u8>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<i16, u8> = DynamicTimeWarping::new(child_metric);
        let x = &[2, 3, 1, 8, 5, 5];
        let y = &[8, 7, 9, 2];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 23);
    }

    #[test]
    fn my_test_4() {
        let child_metric = clam::metric_from_name::<i16, u8>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<i16, u8> = DynamicTimeWarping::new(child_metric);
        let x = &[2, 3, 1, 8];
        let y = &[8, 7, 9, 2];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 23);
    }

    #[test]
    fn my_test_5() {
        let child_metric = clam::metric_from_name::<i16, u8>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<i16, u8> = DynamicTimeWarping::new(child_metric);
        let x = &[2, 3, 1];
        let y = &[8, 7, 9, 2];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 17);
    }

    #[test]
    fn my_test_6() {
        let child_metric = clam::metric_from_name::<i16, u8>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<i16, u8> = DynamicTimeWarping::new(child_metric);
        let x = &[2, 3];
        let y = &[8, 7, 9, 2];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 17);
    }

    #[test]
    fn my_test_7() {
        let child_metric = clam::metric_from_name::<i16, u8>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<i16, u8> = DynamicTimeWarping::new(child_metric);
        let x = &[2];
        let y = &[8, 7, 9, 2];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 18);
    }

    #[test]
    fn my_test_8() {
        let child_metric = clam::metric_from_name::<i16, u8>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<i16, u8> = DynamicTimeWarping::new(child_metric);
        let x = &[2, 3, 1, 8, 5];
        let y = &[8, 7, 9];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 21);
    }

    #[test]
    fn my_test_9() {
        let child_metric = clam::metric_from_name::<f64, f32>("euclidean", false).unwrap();
        let dtw: DynamicTimeWarping<f64, f32> = DynamicTimeWarping::new(child_metric);
        let x = &[3.14, 3.14, 3.14, 3.14, 3.14];
        let y = &[3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 0.0);
    }

    #[test]
    fn my_test_10() {
        let child_metric = clam::metric_from_name::<f64, f32>("manhattan", false).unwrap();
        let dtw: DynamicTimeWarping<f64, f32> = DynamicTimeWarping::new(child_metric);
        let x = &[0.097, 1.419, 0.245, 6.331, 4.276, 3.154, 0.542, 7.115];
        let y = &[0.097, 1.419, 0.245, 6.331, 4.276, 3.154, 0.542, 7.115];
        let cost = dtw.one_to_one(x, y);

        assert!(cost == 0.0);
    }
}
