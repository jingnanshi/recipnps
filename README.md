# recipnps

Simple recipes for Perspective-n-Points (PnP).
Currently includes P3P routines and PnP routines with RANSAC and P3P solvers.
See the companion blog post [here](https://jingnanshi.com/blog/pnp_minimal.html).

## Examples
### P3P
```rust
use recipnps::p3p::{fischler, grunert, kneip};

// p3p with grunert's method
let mut solutions = grunert(&world_points, &bearing_vectors);

// p3p with fischler's method
solutions = fischler(&world_points, &bearing_vectors);

// p3p with kneip's method
solutions = grunert(&world_points, &bearing_vectors);
```

### PnP
```rust
use recipnps::pnp::{pnp_ransac_fischler, pnp_ransac_grunert, pnp_ransac_kneip};

// pnp with grunert's method + RANSAC
let mut solution = pnp_ransac_grunert(&world_points, &bearing_vectors);

// pnp with fischler's method + RANSAC
solution = pnp_ransac_fischler(&world_points, &bearing_vectors);

// pnp with kneip's method + RANSAC
solution = pnp_ransac_kneip(&world_points, &bearing_vectors);
```

See tests and documentation for more examples.

## References
1. B. M. Haralick, C.-N. Lee, K. Ottenberg, and M. Nölle, “Review and analysis of solutions of the three point perspective pose estimation problem,” Int J Comput Vision, vol. 13, no. 3, pp. 331–356, Dec. 1994, doi: 10.1007/BF02028352.
2. L. Kneip, D. Scaramuzza, and R. Siegwart, “A novel parametrization of the perspective-three-point problem for a direct computation of absolute camera position and orientation,” in CVPR 2011, Jun. 2011, pp. 2969–2976. doi: 10.1109/CVPR.2011.5995464.
3. X.-S. Gao, X.-R. Hou, J. Tang, and H.-F. Cheng, “Complete solution classification for the perspective-three-point problem,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 25, no. 8, pp. 930–943, Aug. 2003, doi: 10.1109/TPAMI.2003.1217599.

