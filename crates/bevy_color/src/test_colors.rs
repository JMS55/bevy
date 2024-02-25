// Generated by gen_tests. Do not edit.
#[cfg(test)]
use crate::{Hsla, Lcha, LinearRgba, Oklaba, Srgba, Xyza};

#[cfg(test)]
pub struct TestColor {
    pub name: &'static str,
    pub rgb: Srgba,
    pub linear_rgb: LinearRgba,
    pub hsl: Hsla,
    pub lch: Lcha,
    pub oklab: Oklaba,
    pub xyz: Xyza,
}

// Table of equivalent colors in various color spaces
#[cfg(test)]
pub const TEST_COLORS: &[TestColor] = &[
    // black
    TestColor {
        name: "black",
        rgb: Srgba::new(0.0, 0.0, 0.0, 1.0),
        linear_rgb: LinearRgba::new(0.0, 0.0, 0.0, 1.0),
        hsl: Hsla::new(0.0, 0.0, 0.0, 1.0),
        lch: Lcha::new(0.0, 0.0, 0.0000136603785, 1.0),
        oklab: Oklaba::new(0.0, 0.0, 0.0, 1.0),
        xyz: Xyza::new(0.0, 0.0, 0.0, 1.0),
    },
    // white
    TestColor {
        name: "white",
        rgb: Srgba::new(1.0, 1.0, 1.0, 1.0),
        linear_rgb: LinearRgba::new(1.0, 1.0, 1.0, 1.0),
        hsl: Hsla::new(0.0, 0.0, 1.0, 1.0),
        lch: Lcha::new(1.0, 0.0, 0.0000136603785, 1.0),
        oklab: Oklaba::new(1.0, 0.0, 0.000000059604645, 1.0),
        xyz: Xyza::new(0.95047, 1.0, 1.08883, 1.0),
    },
    // red
    TestColor {
        name: "red",
        rgb: Srgba::new(1.0, 0.0, 0.0, 1.0),
        linear_rgb: LinearRgba::new(1.0, 0.0, 0.0, 1.0),
        hsl: Hsla::new(0.0, 1.0, 0.5, 1.0),
        lch: Lcha::new(0.53240794, 1.0455177, 39.99901, 1.0),
        oklab: Oklaba::new(0.6279554, 0.22486295, 0.1258463, 1.0),
        xyz: Xyza::new(0.4124564, 0.2126729, 0.0193339, 1.0),
    },
    // green
    TestColor {
        name: "green",
        rgb: Srgba::new(0.0, 1.0, 0.0, 1.0),
        linear_rgb: LinearRgba::new(0.0, 1.0, 0.0, 1.0),
        hsl: Hsla::new(120.0, 1.0, 0.5, 1.0),
        lch: Lcha::new(0.87734723, 1.1977587, 136.01595, 1.0),
        oklab: Oklaba::new(0.8664396, -0.2338874, 0.1794985, 1.0),
        xyz: Xyza::new(0.3575761, 0.7151522, 0.119192, 1.0),
    },
    // blue
    TestColor {
        name: "blue",
        rgb: Srgba::new(0.0, 0.0, 1.0, 1.0),
        linear_rgb: LinearRgba::new(0.0, 0.0, 1.0, 1.0),
        hsl: Hsla::new(240.0, 1.0, 0.5, 1.0),
        lch: Lcha::new(0.32297012, 1.3380761, 306.28494, 1.0),
        oklab: Oklaba::new(0.4520137, -0.032456964, -0.31152815, 1.0),
        xyz: Xyza::new(0.1804375, 0.072175, 0.9503041, 1.0),
    },
    // yellow
    TestColor {
        name: "yellow",
        rgb: Srgba::new(1.0, 1.0, 0.0, 1.0),
        linear_rgb: LinearRgba::new(1.0, 1.0, 0.0, 1.0),
        hsl: Hsla::new(60.0, 1.0, 0.5, 1.0),
        lch: Lcha::new(0.9713927, 0.96905375, 102.85126, 1.0),
        oklab: Oklaba::new(0.9679827, -0.07136908, 0.19856972, 1.0),
        xyz: Xyza::new(0.7700325, 0.9278251, 0.1385259, 1.0),
    },
    // magenta
    TestColor {
        name: "magenta",
        rgb: Srgba::new(1.0, 0.0, 1.0, 1.0),
        linear_rgb: LinearRgba::new(1.0, 0.0, 1.0, 1.0),
        hsl: Hsla::new(300.0, 1.0, 0.5, 1.0),
        lch: Lcha::new(0.6032421, 1.1554068, 328.23495, 1.0),
        oklab: Oklaba::new(0.7016738, 0.27456632, -0.16915613, 1.0),
        xyz: Xyza::new(0.5928939, 0.28484792, 0.969638, 1.0),
    },
    // cyan
    TestColor {
        name: "cyan",
        rgb: Srgba::new(0.0, 1.0, 1.0, 1.0),
        linear_rgb: LinearRgba::new(0.0, 1.0, 1.0, 1.0),
        hsl: Hsla::new(180.0, 1.0, 0.5, 1.0),
        lch: Lcha::new(0.9111322, 0.50120866, 196.37614, 1.0),
        oklab: Oklaba::new(0.90539926, -0.1494439, -0.039398134, 1.0),
        xyz: Xyza::new(0.5380136, 0.78732723, 1.069496, 1.0),
    },
    // gray
    TestColor {
        name: "gray",
        rgb: Srgba::new(0.5, 0.5, 0.5, 1.0),
        linear_rgb: LinearRgba::new(0.21404114, 0.21404114, 0.21404114, 1.0),
        hsl: Hsla::new(0.0, 0.0, 0.5, 1.0),
        lch: Lcha::new(0.5338897, 0.00000011920929, 90.0, 1.0),
        oklab: Oklaba::new(0.5981807, 0.00000011920929, 0.0, 1.0),
        xyz: Xyza::new(0.2034397, 0.21404117, 0.23305441, 1.0),
    },
    // olive
    TestColor {
        name: "olive",
        rgb: Srgba::new(0.5, 0.5, 0.0, 1.0),
        linear_rgb: LinearRgba::new(0.21404114, 0.21404114, 0.0, 1.0),
        hsl: Hsla::new(60.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.51677734, 0.57966936, 102.851265, 1.0),
        oklab: Oklaba::new(0.57902855, -0.042691574, 0.11878061, 1.0),
        xyz: Xyza::new(0.16481864, 0.19859275, 0.029650241, 1.0),
    },
    // purple
    TestColor {
        name: "purple",
        rgb: Srgba::new(0.5, 0.0, 0.5, 1.0),
        linear_rgb: LinearRgba::new(0.21404114, 0.0, 0.21404114, 1.0),
        hsl: Hsla::new(300.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.29655674, 0.69114214, 328.23495, 1.0),
        oklab: Oklaba::new(0.41972777, 0.1642403, -0.10118592, 1.0),
        xyz: Xyza::new(0.12690368, 0.060969174, 0.20754242, 1.0),
    },
    // teal
    TestColor {
        name: "teal",
        rgb: Srgba::new(0.0, 0.5, 0.5, 1.0),
        linear_rgb: LinearRgba::new(0.0, 0.21404114, 0.21404114, 1.0),
        hsl: Hsla::new(180.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.48073065, 0.29981336, 196.37614, 1.0),
        oklab: Oklaba::new(0.54159236, -0.08939436, -0.02356726, 1.0),
        xyz: Xyza::new(0.11515705, 0.16852042, 0.22891617, 1.0),
    },
    // maroon
    TestColor {
        name: "maroon",
        rgb: Srgba::new(0.5, 0.0, 0.0, 1.0),
        linear_rgb: LinearRgba::new(0.21404114, 0.0, 0.0, 1.0),
        hsl: Hsla::new(0.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.2541851, 0.61091745, 38.350803, 1.0),
        oklab: Oklaba::new(0.3756308, 0.13450874, 0.07527886, 1.0),
        xyz: Xyza::new(0.08828264, 0.045520753, 0.0041382504, 1.0),
    },
    // lime
    TestColor {
        name: "lime",
        rgb: Srgba::new(0.0, 0.5, 0.0, 1.0),
        linear_rgb: LinearRgba::new(0.0, 0.21404114, 0.0, 1.0),
        hsl: Hsla::new(120.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.46052113, 0.71647626, 136.01596, 1.0),
        oklab: Oklaba::new(0.5182875, -0.13990697, 0.10737252, 1.0),
        xyz: Xyza::new(0.076536, 0.153072, 0.025511991, 1.0),
    },
    // navy
    TestColor {
        name: "navy",
        rgb: Srgba::new(0.0, 0.0, 0.5, 1.0),
        linear_rgb: LinearRgba::new(0.0, 0.0, 0.21404114, 1.0),
        hsl: Hsla::new(240.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.12890343, 0.8004114, 306.28494, 1.0),
        oklab: Oklaba::new(0.27038592, -0.01941514, -0.18635012, 1.0),
        xyz: Xyza::new(0.03862105, 0.01544842, 0.20340417, 1.0),
    },
    // orange
    TestColor {
        name: "orange",
        rgb: Srgba::new(0.5, 0.5, 0.0, 1.0),
        linear_rgb: LinearRgba::new(0.21404114, 0.21404114, 0.0, 1.0),
        hsl: Hsla::new(60.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.51677734, 0.57966936, 102.851265, 1.0),
        oklab: Oklaba::new(0.57902855, -0.042691574, 0.11878061, 1.0),
        xyz: Xyza::new(0.16481864, 0.19859275, 0.029650241, 1.0),
    },
    // fuchsia
    TestColor {
        name: "fuchsia",
        rgb: Srgba::new(0.5, 0.0, 0.5, 1.0),
        linear_rgb: LinearRgba::new(0.21404114, 0.0, 0.21404114, 1.0),
        hsl: Hsla::new(300.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.29655674, 0.69114214, 328.23495, 1.0),
        oklab: Oklaba::new(0.41972777, 0.1642403, -0.10118592, 1.0),
        xyz: Xyza::new(0.12690368, 0.060969174, 0.20754242, 1.0),
    },
    // aqua
    TestColor {
        name: "aqua",
        rgb: Srgba::new(0.0, 0.5, 0.5, 1.0),
        linear_rgb: LinearRgba::new(0.0, 0.21404114, 0.21404114, 1.0),
        hsl: Hsla::new(180.0, 1.0, 0.25, 1.0),
        lch: Lcha::new(0.48073065, 0.29981336, 196.37614, 1.0),
        oklab: Oklaba::new(0.54159236, -0.08939436, -0.02356726, 1.0),
        xyz: Xyza::new(0.11515705, 0.16852042, 0.22891617, 1.0),
    },
];
