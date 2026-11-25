# Perfusion parameters for the mouse liver lobule (milestone 2)

Status: 2026-11-25. Used by the end-to-end study of milestone 2
(`docs/superpowers/specs/2026-11-25-slidesim-m2-perfusion-design.md`, verification item 6).

## Scope and method

The slides of slideviz are mostly mouse liver (acetaminophen studies), so mouse values come first.
Where no mouse value was found, rat, then human values are used, and the species is stated.

Every value below was taken from a source that was opened (abstract or full text), at the place
given. Values seen only in search result snippets are marked **unverified** and are not used for
the baseline. Two values were checked a second time against the PubMed abstract: the Debbaut 2012
permeabilities and porosity, and the Xie 2014 portal pressure. The Tsukada 2013 numbers come from
its figures (Fig. 1C, 1D) and were not checked a second time.

Model units: um, s, Pa, um^2, Pa*s, mM.

## Baseline parameter set

| Parameter | Study key | Value | Species | Source |
|---|---|---|---|---|
| Pressure drop portal to central | `perfusion.boundaries` | 390 Pa (`portal_wall` 390 Pa, `central_wall` 0 Pa) | rat, human | [Shibayama 1975], [Siggers 2014] Table 1 |
| Viscosity | `perfusion.viscosity` | 2.4e-3 Pa*s | human (formula) | [Siggers 2014] Table 1 |
| Permeability, radial | `perfusion.k_along` | 0.0156 um^2 | human | [Debbaut 2012] abstract |
| Permeability, circumferential | `perfusion.k_across` | 0.0175 um^2 | human | [Debbaut 2012] abstract |
| Isotropy scale | `perfusion.direction.g0` | 4.3e-4 1/um (= 0.1/L) | derived | this document |
| Porosity | field `porosity` | 0.15 | mouse, human | [Hammad 2014], [Debbaut 2012] |
| Tracer diffusion (sucrose) | field `diffusion` | 700 um^2/s | water, 37 C (scaled) | [BioNumbers 100614] |
| Portal to central distance | (check only) | L = 230 um | mouse | [Tsukada 2013] Fig. 1C |

All coefficients are constants in the baseline (no tables). The zonation map is still used: it
gives the direction of the anisotropy.

## Parameter notes

**Pressures.** The only mouse portal pressure found is 6.87 +/- 2.39 cmH2O (674 Pa) by catheter
[Xie 2014]. No mouse central vein or sinusoidal pressure was found. In the rat, micropuncture gives
68 mmH2O in the terminal portal venule and 28 mmH2O in the central vein [Shibayama 1975], a drop of
40 mmH2O = 392 Pa. The human lobule model of [Bonfiglio 2010], as cited in [Siggers 2014] Table 1,
uses 4.4 and 1.5 mmHg, a drop of 387 Pa. Both agree at about 390 Pa. Only the drop matters for
the flow, so the central wall is set to 0 Pa.

**Viscosity.** The apparent viscosity of blood in vessels of 5-10 um is below that of whole blood
(Fahraeus-Lindqvist effect). [Siggers 2014] uses 1.33 x plasma viscosity (1.8 mPa*s), that is
2.4 mPa*s, from the in vitro formula of Pries et al. at a diameter of 10 um. [Debbaut 2012, 2014]
use whole blood, 3.5 mPa*s. Range 1.5-3.5 mPa*s.

**Permeability.** Only human data exist. [Debbaut 2012] computed the permeability tensor from
micro-CT of a human lobule: radial 1.56e-14 m^2, circumferential 1.75e-14 m^2, along the central
vein 3.64e-14 m^2. The radial and circumferential values are about equal (ratio 0.89): in the
plane of a section the sinusoid network is close to isotropic. The strong anisotropy is along the
central vein axis, which a 2D section does not contain. [Bonfiglio 2010] reports that anisotropy
has a small quantitative effect. No mouse permeability was found.

Consequence for milestone 2: the anisotropic permeability along the zonation gradient stays in the
design (it is the right mechanism, and the direction map is also the input for a later measured
orientation), but the literature baseline is nearly isotropic. A strongly radial ratio
(k_along >> k_across) is not supported by data and is used only as a sensitivity case.

**Porosity.** Mouse sinusoidal volume fraction 15.3 +/- 3.9 % [Hammad 2014]; human 3D porosity
14.3 % [Debbaut 2012]. 0.15 is used. Flag: [Hammad 2014]'s own length density with its sinusoid
radius implies a much larger volume fraction, so the mouse value is uncertain; the human value
supports 0.14-0.15.

**Geometry.** The sinusoid length between portal and central venules measured intravitally in
control mice is 232 +/- 60 um [Tsukada 2013] Fig. 1C, used as L = 230 um. On a real slide the
distances come from the mesh; L is used only for the checks below.

**Tracer.** Sucrose, an inert small tracer. 520 um^2/s in water [BioNumbers 100614], temperature
not stated; scaled to 37 C by T/viscosity (factor 1.34): about 700 um^2/s. Acetaminophen: 660
um^2/s at 25 C [Ribeiro 2012], about 880 um^2/s at 37 C by the same scaling. Plasma lowers both by
an estimated 10-20 % (no source). 4 kDa FITC-dextran (about 157 um^2/s) and inulin (150 um^2/s)
are **unverified** and not used.

## Experimental data for comparison

| Quantity | Value | Species | Source |
|---|---|---|---|
| Red blood cell velocity in sinusoids | 180 +/- 60 um/s | mouse (intravital, pentobarbital + isoflurane) | [Tsukada 2013] Fig. 1D |
| Red blood cell velocity in sinusoids | 69.2 +/- 30.6 um/s | mouse BALB/cJ | [MacPhee 1988] abstract |
| Flow pattern by zone | fast flow 76 % of the time in zone 3 vs 51 % in zone 1; stasis 19 % vs 2 % | mouse | [MacPhee 1995] abstract |
| Portal / hepatic artery flow | 1.6-2.3 / 0.10-0.35 mL/min | mouse | [Xie 2014] |

The two mouse velocities differ by 2.6x (strain, anesthesia, method). Intravital microscopy sees
superficial, flowing sinusoids and probably overestimates the mean. No measured sinusoidal transit
time was found; transit times below are derived.

## Consistency check

With the baseline (k = 0.0156 um^2, mu = 2.4e-3 Pa*s, dP = 390 Pa, L = 230 um, phi = 0.15):

- Darcy velocity u = k/mu * dP/L = 6.5 um^2/(Pa*s) * 1.70 Pa/um = 11.0 um/s
- interstitial velocity u/phi = 73 um/s
- transit time L/(u/phi) = 3.1 s

73 um/s matches [MacPhee 1988] (69 um/s) and is 2.5x below [Tsukada 2013] (180 um/s). With whole
blood viscosity (3.5 mPa*s): 50 um/s, transit 4.6 s.

Lobule-scale Peclet number with the interstitial velocity, (u/phi) L / D: sucrose 24, acetaminophen
19. Diffusion time L^2/D: 76 s (sucrose). So the tracer front is carried by the flow: it reaches
the central walls after about 3 s and is spread by diffusion over the following tens of seconds.

Cell Peclet number with the Darcy velocity, u h / D: sucrose 0.08 at h = 5 um and 0.31 at h = 20
um. The real-slide run is diffusion-dominated at the cell scale; the advection-dominated SUPG tests
(cell Peclet about 100) use manufactured solutions, not these parameters.

## Isotropy scale g0

|grad p| of the zonation map is about 1/L = 4.3e-3 1/um. With g0 = 0.1/L = 4.3e-4 1/um the weight
w = |g|^2/(|g|^2 + g0^2) is above 0.99 wherever |g| >= 1/L; the blend to isotropic acts only at
peaks, valleys and lobule borders, where the direction is undefined. With the baseline ratio 0.89
the choice of g0 changes the result very little.

## Sensitivity cases

1. **Calibrated to the fast velocity**: k_along chosen so that u/phi = 180 um/s [Tsukada 2013]:
   k_along = v phi mu L / dP = 0.038 um^2 (2.45 x literature), k_across = k_along / 0.89 =
   0.043 um^2. A 2D section misses the axial flow (axial k is 2.3 x radial), which is one reason a
   larger in-plane k can be justified.
2. **Whole blood viscosity**: mu = 3.5e-3 Pa*s.
3. **Strong radial anisotropy** (not supported by data, tests the mechanism): k_across =
   k_along / 10.

## References

- [BioNumbers 100614] Diffusion coefficient of sucrose in water, from Vogel S, Life's Devices,
  1989. https://bionumbers.hms.harvard.edu/bionumber.aspx?id=100614
- [Bonfiglio 2010] Bonfiglio A et al. Mathematical modeling of the circulation in the liver lobule.
  J Biomech Eng 2010. DOI 10.1115/1.4002563, PMID 21034152 (abstract; values via [Siggers 2014]).
- [Debbaut 2012] Debbaut C et al. Perfusion characteristics of the human hepatic microcirculation
  based on three-dimensional reconstructions and computational fluid dynamic analysis. J Biomech
  Eng 2012;134:011003. DOI 10.1115/1.4005545, PMID 22482658.
- [Debbaut 2014] Debbaut C et al. A 3D porous media liver lobule model: the importance of vascular
  septa and anisotropic permeability for homogeneous perfusion. Comput Methods Biomech Biomed Eng
  2014;17:1295. DOI 10.1080/10255842.2012.744399, PMID 23237543.
- [Hammad 2014] Hammad S et al. Protocols for staining of bile canalicular and sinusoidal networks
  of human, mouse and pig livers, three-dimensional reconstruction and quantification of tissue
  microarchitecture by image processing and analysis. Arch Toxicol 2014;88:1161. DOI
  10.1007/s00204-014-1243-5, PMID 24748404.
- [MacPhee 1988] MacPhee PJ et al. Microcirculatory changes in livers of mice infected with murine
  hepatitis virus. Evidence from microcorrosion casts and measurements of red cell velocity.
  Microvasc Res 1988;36:140. DOI 10.1016/0026-2862(88)90014-3,
  PMID 2846988.
- [MacPhee 1995] MacPhee PJ et al. Intermittence of blood flow in liver sinusoids, studied by
  high-resolution in vivo microscopy. Am J Physiol 1995;269:G692. DOI
  10.1152/ajpgi.1995.269.5.G692, PMID 7491960.
- [Ribeiro 2012] Ribeiro ACF et al. J Chem Thermodyn 2012;54:97. DOI 10.1016/j.jct.2012.03.014.
- [Shibayama 1975] Shibayama Y. Kanzo 1975;16:145. DOI 10.2957/kanzo.16.145.
- [Siggers 2014] Siggers JH et al. Mathematical model of blood and interstitial flow and lymph
  production in the liver. Biomech Model Mechanobiol 2014. DOI 10.1007/s10237-013-0516-x,
  PMID 23907149.
- [Tsukada 2013] Tsukada K et al. Hypoxia-inducible factor-1 is a determinant of lobular structure
  and oxygen consumption in the liver. Microcirculation 2013;20:385. DOI 10.1111/micc.12033,
  PMC6592815.
- [Xie 2014] Xie C et al. Monitoring of systemic and hepatic hemodynamic parameters in mice. J Vis
  Exp 2014. DOI 10.3791/51955, PMID 25350047.

## Gaps

- No mouse measurements of central vein or sinusoidal pressure, or of lobule permeability.
- No zone-resolved mouse sinusoidal velocities as numbers (zonal porosity exists as a figure in
  Morales-Navarrete et al., eLife 2019, DOI 10.7554/eLife.44860, not extracted).
- No diffusion coefficients measured in plasma.
