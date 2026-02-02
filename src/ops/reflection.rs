use crate::{bivector::Bivector3, ga::geometric_product_full, vector::Vec3};

/// Simple scalar‐formula reflection
pub trait Vec3Reflect {
    fn reflect_in_plane(self, normal: Vec3) -> Vec3;
}

impl Vec3Reflect for Vec3 {
    #[inline(always)]
    fn reflect_in_plane(self, normal: Vec3) -> Vec3 {
        let denom = normal.dot(&normal);
        if denom == 0.0 {
            return self;
        }
        self - normal * (2.0 * self.dot(&normal) / denom)
    }
}

/// GA‐based reflection using a pure bivector `b` (unit magnitude)
pub fn reflect_ga(v: Vec3, b: Bivector3) -> Vec3 {
    let mut v_mv = [0.0; 8];
    v_mv[1] = v.x;
    v_mv[2] = v.y;
    v_mv[3] = v.z;
    let mut b_mv = [0.0; 8];
    b_mv[4] = b.xy;
    b_mv[5] = b.yz;
    b_mv[6] = b.zx;
    let mut tmp = [0.0; 8];
    let mut out = [0.0; 8];
    // b * v
    geometric_product_full(&b_mv, &v_mv, &mut tmp);
    // (b * v) * (–b)
    let mut b_neg = b_mv;
    b_neg[4..7].iter_mut().for_each(|x| *x = -*x);
    geometric_product_full(&tmp, &b_neg, &mut out);
    Vec3::new(out[1], out[2], out[3])
}
